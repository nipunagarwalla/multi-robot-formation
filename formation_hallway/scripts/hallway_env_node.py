#!/usr/bin/python3
"""
The Gym-shaped wrapper around the Gazebo hallway. Loads a trained
FormationHallwayEnv PPO policy and drives the 4 Tritons.

Why this exists:
  multi-robot-formation/code/env_hallway.py exposes an obs dict with
  pos / vel / goal / teleop_mask / present_mask / time, and an action of
  per-robot 2D world-frame velocity. The policy is trained against that
  exact contract. This node mirrors it on top of Gazebo:

    1. Subscribes to /triton_<i>/odom (4x) for ground-truth pose + twist
    2. Subscribes to /teleop_mask for the toggle teleop state
    3. Each tick (20 Hz):
         - assembles obs dict from latest odom
         - calls agent.format_input(...) and get_action_and_value(...)
         - for each non-teleop robot, rotates world->body and publishes
           /triton_<i>/cmd_vel (the model_push plugin expects body-frame)
         - tracks episode time, success on cluster centroid >= GOAL_Y
    4. On episode end (timeout or goal), teleports robots back to spawn
       via /gazebo/set_model_state and zeroes velocities.

Usage:
  rosrun formation_hallway hallway_env_node.py \
      --weights /path/to/runs/<ts>/weights/latest.pt \
      --policy-repo /Users/nipun/Desktop/multi-robot-formation
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from std_msgs.msg import Float32MultiArray

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "config"))
from contract import (  # noqa: E402
    DT, FORMATION_SCALE, GOAL_Y, MAX_AGENTS, MAX_V,
    SPAWN_Y, WORLD_H, WORLD_W,
)


# -------------------------------------------------------------------- math --
def yaw_from_quat(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def square_slots(scale: float):
    s = scale
    return [(-s/2, -s/2), (s/2, -s/2), (s/2, s/2), (-s/2, s/2)]


# ---------------------------------------------------------------- env class --
class GazeboHallwayEnv:
    """A drop-in shim that quacks enough like FormationHallwayEnv for the
    policy's Agent wrapper to consume. Specifically, Agent reads
    `env.observation_space` and `env.action_space` at construction and
    nothing else."""

    def __init__(self):
        import gymnasium as gym
        n = MAX_AGENTS
        self.cfg = {
            "n_agents": n,
            "num_envs": 1,
            "dt": DT,
            "max_v": MAX_V,
        }
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Box(low=-MAX_V, high=MAX_V, shape=(2,), dtype=float),) * n
        )
        max_t = 600 * DT
        self.observation_space = gym.spaces.Dict({
            "pos":          gym.spaces.Box(-WORLD_H, WORLD_H, shape=(n, 2), dtype=float),
            "vel":          gym.spaces.Box(-1e5, 1e5, shape=(n, 2), dtype=float),
            "goal":         gym.spaces.Box(-WORLD_H, WORLD_H, shape=(n, 2), dtype=float),
            "teleop_mask":  gym.spaces.Box(0.0, 1.0, shape=(n,), dtype=float),
            "present_mask": gym.spaces.Box(0.0, 1.0, shape=(n,), dtype=float),
            "time":         gym.spaces.Box(0.0, max_t, shape=(n, 1), dtype=float),
        })


# ------------------------------------------------------------------ ros side --
class HallwayPolicyRunner:
    def __init__(self, weights_path: str, policy_repo: str, device: str = "cpu",
                 max_steps: int = 600, k_yaw: float = 1.0):
        # make the policy code importable
        code_dir = os.path.join(policy_repo, "code")
        if not os.path.isdir(code_dir):
            raise RuntimeError(f"policy repo not found: {code_dir}")
        sys.path.insert(0, code_dir)

        import torch
        from checkpoint import load_checkpoint
        from model import Agent
        self.torch = torch

        self.device = torch.device(device)
        self.env_shim = GazeboHallwayEnv()
        agent_cfg = {
            "model": {
                "custom_model_config": {
                    "activation": "relu",
                    "msg_features": 32,
                    "comm_range": 2.0,
                    "use_masks": True,
                }
            }
        }
        self.agent = Agent(self.env_shim, agent_cfg).to(self.device)
        ckpt = load_checkpoint(weights_path, self.device)
        self.agent.load_state_dict(ckpt["agent"])
        self.agent.eval()
        rospy.loginfo(f"[env] loaded weights: {weights_path} (iter={ckpt.get('iteration')})")

        self.n = MAX_AGENTS
        self.max_steps = max_steps
        self.k_yaw = k_yaw

        # latest odom snapshot per robot
        self.poses_xy = np.zeros((self.n, 2), dtype=np.float32)
        self.vels_xy = np.zeros((self.n, 2), dtype=np.float32)
        self.yaws = np.zeros(self.n, dtype=np.float32)
        self.have_odom = [False] * self.n

        self.teleop_mask = np.zeros(self.n, dtype=np.float32)
        self.present_mask = np.ones(self.n, dtype=np.float32)
        self.t = 0.0
        self.step_count = 0
        self.goal_reached = False
        self.episodes = 0

        self.cmd_pubs = [
            rospy.Publisher(f"/triton_{i+1}/cmd_vel", Twist, queue_size=1)
            for i in range(self.n)
        ]
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._model_states_cb, queue_size=1)
        rospy.Subscriber("/teleop_mask", Float32MultiArray,
                         self._teleop_cb, queue_size=1)

        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        self._lock = threading.Lock()

    # -------------------------------------------------------------- callbacks
    def _model_states_cb(self, msg: ModelStates):
        with self._lock:
            for i in range(self.n):
                name = f"triton_{i+1}"
                try:
                    idx = msg.name.index(name)
                except ValueError:
                    continue
                p = msg.pose[idx].position
                v = msg.twist[idx].linear
                q = msg.pose[idx].orientation
                self.poses_xy[i, 0] = p.x
                self.poses_xy[i, 1] = p.y
                self.vels_xy[i, 0] = v.x
                self.vels_xy[i, 1] = v.y
                self.yaws[i] = yaw_from_quat(q.x, q.y, q.z, q.w)
                self.have_odom[i] = True

    def _teleop_cb(self, msg: Float32MultiArray):
        if len(msg.data) != self.n:
            return
        with self._lock:
            self.teleop_mask = np.asarray(msg.data, dtype=np.float32)

    # -------------------------------------------------------------- obs
    def _build_obs(self):
        # mirrors FormationHallwayEnv.get_obs(0): all lists, broadcast goal/time
        obs = {
            "pos":          self.poses_xy.tolist(),
            "vel":          self.vels_xy.tolist(),
            "goal":         [[0.0, GOAL_Y]] * self.n,
            "teleop_mask":  self.teleop_mask.tolist(),
            "present_mask": self.present_mask.tolist(),
            "time":         [[self.t]] * self.n,
        }
        return obs

    # -------------------------------------------------------------- reset
    def _reset_episode(self):
        rospy.loginfo(f"[env] resetting episode (was step={self.step_count}, "
                      f"goal={self.goal_reached})")
        # 1. zero cmd_vel for all (defeat plugin's 0.15 s last-command memory)
        for p in self.cmd_pubs:
            p.publish(Twist())
        rospy.sleep(0.06)

        # 2. teleport each robot to a jittered slot near (0, SPAWN_Y), yaw=0
        rng = np.random.default_rng()
        slots = square_slots(FORMATION_SCALE)
        for i in range(self.n):
            sx, sy = slots[i]
            jx = rng.uniform(-0.05, 0.05)
            jy = rng.uniform(-0.05, 0.05)
            ms = ModelState()
            ms.model_name = f"triton_{i+1}"
            ms.reference_frame = "world"
            ms.pose.position.x = float(sx + jx)
            ms.pose.position.y = float(SPAWN_Y + sy + jy)
            ms.pose.position.z = 0.0
            ms.pose.orientation.w = 1.0  # yaw=0
            # zero twist
            try:
                self.set_state(ms)
            except rospy.ServiceException as e:
                rospy.logwarn(f"[env] set_model_state failed for triton_{i+1}: {e}")
        rospy.sleep(0.10)

        with self._lock:
            self.t = 0.0
            self.step_count = 0
            self.goal_reached = False
            self.episodes += 1

    # -------------------------------------------------------------- step
    def _publish_action(self, action_world):
        """action_world: (n, 2) numpy in world frame."""
        with self._lock:
            mask = self.teleop_mask.copy()
            yaws = self.yaws.copy()
        for i in range(self.n):
            if mask[i] > 0.5:
                continue  # teleop node owns this one
            vx_w, vy_w = float(action_world[i, 0]), float(action_world[i, 1])
            yaw = float(yaws[i])
            tw = Twist()
            # rotate world->body for the model_push plugin
            tw.linear.x =  vx_w * math.cos(yaw) + vy_w * math.sin(yaw)
            tw.linear.y = -vx_w * math.sin(yaw) + vy_w * math.cos(yaw)
            # passive yaw damping toward 0 — keeps body~world frames aligned
            tw.angular.z = -self.k_yaw * yaw
            self.cmd_pubs[i].publish(tw)

    def _check_done(self):
        # goal: cluster centroid (active subset) crosses GOAL_Y
        with self._lock:
            mask = self.teleop_mask.copy()
            poses = self.poses_xy.copy()
        active = np.where(mask < 0.5)[0]
        if len(active) > 0:
            centroid_y = float(poses[active, 1].mean())
            if centroid_y >= GOAL_Y and not self.goal_reached:
                self.goal_reached = True
                rospy.loginfo(f"[env] GOAL reached at step {self.step_count} "
                              f"(centroid_y={centroid_y:+.3f})")
        timeout = self.step_count >= self.max_steps
        return self.goal_reached or timeout

    # -------------------------------------------------------------- main loop
    def run(self, autoreset: bool = True):
        torch = self.torch
        # wait for first odom from all robots
        rospy.loginfo("[env] waiting for odom from all robots...")
        deadline = rospy.Time.now() + rospy.Duration(10.0)
        while not all(self.have_odom) and rospy.Time.now() < deadline:
            rospy.sleep(0.1)
        if not all(self.have_odom):
            rospy.logwarn(f"[env] odom missing: {self.have_odom}; continuing anyway")

        self._reset_episode()

        rate = rospy.Rate(int(round(1.0 / DT)))
        while not rospy.is_shutdown():
            obs = self._build_obs()

            with torch.no_grad():
                x = self.agent.format_input([obs], self.device)
                action, _, _, _ = self.agent.get_action_and_value(x)
            action_np = action[0].cpu().numpy()  # (n, 2)
            # safety clip in case the dist samples slightly out of bound
            action_np = np.clip(action_np, -MAX_V, MAX_V)

            self._publish_action(action_np)

            with self._lock:
                self.step_count += 1
                self.t += DT

            if self._check_done():
                if not autoreset:
                    rospy.loginfo("[env] episode done, --no-autoreset; exiting")
                    break
                self._reset_episode()

            rate.sleep()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to a .pt checkpoint")
    ap.add_argument("--policy-repo", required=True,
                    help="path to multi-robot-formation/ (containing code/)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--k-yaw", type=float, default=1.0,
                    help="proportional gain damping yaw toward 0")
    ap.add_argument("--no-autoreset", action="store_true")
    args = ap.parse_args(rospy.myargv()[1:])

    rospy.init_node("hallway_env_node")
    runner = HallwayPolicyRunner(
        weights_path=os.path.abspath(args.weights),
        policy_repo=os.path.abspath(args.policy_repo),
        device=args.device,
        max_steps=args.max_steps,
        k_yaw=args.k_yaw,
    )
    runner.run(autoreset=not args.no_autoreset)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        sys.exit(0)
