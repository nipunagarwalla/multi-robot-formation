#!/usr/bin/env python3
"""
ROS2 wrapper around circle_policy_v1 (FormationHallwayEnv PPO).

Mirrors the obs/action contract from code/contract.py + code/env_hallway.py:

  obs = {
    "pos":          (MAX_AGENTS, 2)  world-frame meters
    "vel":          (MAX_AGENTS, 2)  world-frame m/s
    "goal":         (MAX_AGENTS, 2)  broadcast [0, GOAL_Y]
    "teleop_mask":  (MAX_AGENTS,)    {0., 1.}
    "present_mask": (MAX_AGENTS,)    {0., 1.}
    "time":         (MAX_AGENTS, 1)  broadcast t
  }
  action = (MAX_AGENTS, 2) world-frame velocities, ±MAX_V

Single observation source: /model_states (published by gazebo_ros_state).
Single action sink:        /limo_<i>/cmd_vel (geometry_msgs/Twist).

The world-frame (vx, vy) emitted by the policy is rotated into the
robot's body frame before publishing, plus a passive yaw damper, so the
LIMO's stock gazebo_ros_diff_drive plugin can execute it. LIMO ignores
linear.y; see plan risk #1.

Teleop integration (per code/teleop.py:KeyboardTeleop, ported to ROS2):
  /teleop/cmd      Float32MultiArray, length 2*MAX_AGENTS  -- (vx,vy) per slot in world frame
  /teleop/mask     Float32MultiArray, length MAX_AGENTS    -- live teleop mask
  /teleop/present  Float32MultiArray, length MAX_AGENTS    -- live present mask
                                                              (spawn/delete events)

Spawn/delete are realized by listening to /teleop/present: when a slot
flips 0->1 we teleport that LIMO model from the sentinel to the active
centroid (mirrors env_hallway.spawn); 1->0 sends it back to the sentinel.
"""
from __future__ import annotations

import math
import os
import pathlib
import sys
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.msg import ModelStates, ModelState, EntityState
from gazebo_msgs.srv import SetEntityState

# ----------------------------------------------------------------- code/ shim
# Make multi-robot-formation/code/ importable. This file lives at
# multi-robot-formation/ros2/limo_circle_sim/limo_circle_sim/circle_node.py
# so the repo root is 4 parents up.
_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[3]
_CODE = _REPO / "code"
if not _CODE.is_dir():
    raise RuntimeError(f"could not find policy code at {_CODE}")
sys.path.insert(0, str(_CODE))

import torch  # noqa: E402  (after sys.path tweak)
from contract import (  # noqa: E402
    AGENT_RADIUS, DT, GOAL_Y, MAX_AGENTS, MAX_V, SPAWN_Y,
    SENTINEL_X, SENTINEL_Y, WORLD_H, WORLD_W,
)
from model import Agent  # noqa: E402
from checkpoint import load_checkpoint  # noqa: E402
from env_hallway import target_formation_positions  # noqa: E402


# ------------------------------------------------------------------ math --
def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# -------------------------------------------------------------- env shim --
class GazeboHallwayEnv:
    """Quacks like FormationHallwayEnv for Agent.__init__ — provides only
    observation_space and action_space at MAX_AGENTS=10. No physics."""

    def __init__(self):
        import gymnasium as gym
        n = MAX_AGENTS
        self.cfg = {"n_agents": n, "num_envs": 1, "dt": DT, "max_v": MAX_V}
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Box(low=-MAX_V, high=MAX_V, shape=(2,), dtype=float),) * n
        )
        max_t = 600 * DT
        self.observation_space = gym.spaces.Dict({
            "pos":          gym.spaces.Box(-WORLD_H, WORLD_H, shape=(n, 2), dtype=float),
            "vel":          gym.spaces.Box(-1e5, 1e5,         shape=(n, 2), dtype=float),
            "goal":         gym.spaces.Box(-WORLD_H, WORLD_H, shape=(n, 2), dtype=float),
            "teleop_mask":  gym.spaces.Box(0.0, 1.0,          shape=(n,),   dtype=float),
            "present_mask": gym.spaces.Box(0.0, 1.0,          shape=(n,),   dtype=float),
            "time":         gym.spaces.Box(0.0, max_t,        shape=(n, 1), dtype=float),
        })


# ----------------------------------------------------------------- node --
class CircleNode(Node):
    def __init__(self) -> None:
        super().__init__("circle_node")

        # ---- params -----------------------------------------------------
        self.declare_parameter("weights", "")
        self.declare_parameter("num_agents", 6)
        self.declare_parameter("max_agents", MAX_AGENTS)
        self.declare_parameter("device", "cpu")
        # k_yaw=0 disables the yaw damper. The ROS1 reference used k_yaw=1.0
        # because Triton was holonomic — keeping yaw at 0 made the policy's
        # world-frame (vx, vy) directly executable via body-frame linear.y.
        # LIMO is diff-drive, ignores linear.y, and we deliberately spawn
        # yawed +pi/2 so body-+X aligns with world-+Y (the goal direction).
        # A damper toward yaw=0 (or any fixed yaw) actively fights the
        # spawn pose, spinning every robot from t=0. Keep it off until we
        # have a real heading controller (turn-toward-velocity-vector).
        self.declare_parameter("k_yaw", 0.0)
        self.declare_parameter("max_steps", 600)
        self.declare_parameter("autoreset", True)
        self.declare_parameter("entity_prefix", "limo_")
        # Set this to match the launch's `total_robots` arg (defaults to
        # num_agents). The node will only attempt to teleport entities up
        # to this index — silences "entity [limo_N] does not exist" errors
        # when running with total_robots < max_agents.
        self.declare_parameter("total_robots", 0)

        weights = str(self.get_parameter("weights").value)
        if not weights or not os.path.isfile(weights):
            raise RuntimeError(
                f"param `weights` must point at a .pt checkpoint; got {weights!r}"
            )
        self.num_agents = int(self.get_parameter("num_agents").value)
        self.max_agents = int(self.get_parameter("max_agents").value)
        if self.max_agents != MAX_AGENTS:
            raise RuntimeError(
                f"max_agents={self.max_agents} != contract.MAX_AGENTS={MAX_AGENTS}; "
                "the policy buffers are sized for MAX_AGENTS — refusing to load."
            )
        if not (1 <= self.num_agents <= self.max_agents):
            raise RuntimeError(
                f"num_agents={self.num_agents} not in [1, {self.max_agents}]"
            )
        self.device = torch.device(str(self.get_parameter("device").value))
        self.k_yaw = float(self.get_parameter("k_yaw").value)
        self.max_steps = int(self.get_parameter("max_steps").value)
        self.autoreset = bool(self.get_parameter("autoreset").value)
        self.entity_prefix = str(self.get_parameter("entity_prefix").value)
        # total_robots=0 means "trust num_agents and don't touch sentinels"
        # (matches the default launch behavior of spawning only num_agents).
        tr = int(self.get_parameter("total_robots").value)
        self.total_robots = tr if tr > 0 else self.num_agents
        if not (self.num_agents <= self.total_robots <= self.max_agents):
            raise RuntimeError(
                f"total_robots={self.total_robots} must be in "
                f"[num_agents={self.num_agents}, max_agents={self.max_agents}]"
            )

        # ---- load policy ------------------------------------------------
        env_shim = GazeboHallwayEnv()
        agent_cfg = {
            "model": {
                "custom_model_config": {
                    "activation": "relu",
                    "msg_features": 32,
                    "comm_range": 4.0,
                    "use_masks": True,
                }
            }
        }
        self.agent = Agent(env_shim, agent_cfg).to(self.device)
        ckpt = load_checkpoint(weights, self.device)
        # strict=False so legacy checkpoints (missing the masks-input weights) still load
        self.agent.load_state_dict(ckpt["agent"], strict=False)
        self.agent.eval()
        self.get_logger().info(
            f"loaded weights={weights} (iter={ckpt.get('iteration', 0)})"
        )

        # ---- state ------------------------------------------------------
        self.n = MAX_AGENTS
        self._lock = threading.Lock()

        # per-robot world-frame state, updated by /model_states callback
        self.poses_xy = np.zeros((self.n, 2), dtype=np.float32)
        self.vels_xy = np.zeros((self.n, 2), dtype=np.float32)
        self.yaws = np.zeros(self.n, dtype=np.float32)
        self.have_state = [False] * self.n

        # initial masks: first num_agents present, none teleop'd
        self.present_mask = np.zeros(self.n, dtype=np.float32)
        self.present_mask[: self.num_agents] = 1.0
        self.teleop_mask = np.zeros(self.n, dtype=np.float32)
        self.teleop_vels = np.zeros((self.n, 2), dtype=np.float32)

        # last received present mask from teleop node, used to detect edges
        self._last_teleop_present = self.present_mask.copy()

        self.t = 0.0
        self.step_count = 0
        self.goal_reached = False
        self.episodes = 0

        # ---- ROS i/o ----------------------------------------------------
        qos_rt = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.cmd_pubs = [
            self.create_publisher(Twist, f"/{self.entity_prefix}{i+1}/cmd_vel", qos_rt)
            for i in range(self.n)
        ]

        self.create_subscription(
            ModelStates, "/model_states", self._model_states_cb, qos_rt
        )
        self.create_subscription(
            Float32MultiArray, "/teleop/cmd", self._teleop_cmd_cb, qos_rt
        )
        self.create_subscription(
            Float32MultiArray, "/teleop/mask", self._teleop_mask_cb, qos_rt
        )
        self.create_subscription(
            Float32MultiArray, "/teleop/present", self._teleop_present_cb, qos_rt
        )

        # /set_entity_state for episode reset + spawn/delete teleport
        self.set_state = self.create_client(SetEntityState, "/set_entity_state")
        if not self.set_state.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn(
                "/set_entity_state not available — episode reset and spawn/delete "
                "teleports will fail. Did the world load with gazebo_ros_state?"
            )

        # ---- prime the episode and start ticking -----------------------
        self._reset_episode()
        self.timer = self.create_timer(DT, self._tick)
        self.get_logger().info(
            f"circle_node up: max_agents={self.n} initial_n={self.num_agents} "
            f"dt={DT}s ({int(round(1.0 / DT))} Hz)"
        )

    # ------------------------------------------------------ callbacks --
    def _model_states_cb(self, msg: ModelStates) -> None:
        with self._lock:
            for i in range(self.n):
                name = f"{self.entity_prefix}{i+1}"
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
                self.have_state[i] = True

    def _teleop_cmd_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != 2 * self.n:
            return
        with self._lock:
            self.teleop_vels = np.asarray(msg.data, dtype=np.float32).reshape(self.n, 2)

    def _teleop_mask_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != self.n:
            return
        with self._lock:
            self.teleop_mask = np.asarray(msg.data, dtype=np.float32)

    def _teleop_present_cb(self, msg: Float32MultiArray) -> None:
        """Incoming present_mask from the teleop node — spawn/delete edges."""
        if len(msg.data) != self.n:
            return
        new_present = np.asarray(msg.data, dtype=np.float32)
        with self._lock:
            old = self._last_teleop_present.copy()
            self._last_teleop_present = new_present.copy()
            # commit immediately so the policy sees the updated mask
            self.present_mask = new_present.copy()
            # also force teleop off for any slot that just became non-present
            became_absent = (old > 0.5) & (new_present < 0.5)
            self.teleop_mask[became_absent] = 0.0
            self.teleop_vels[became_absent] = 0.0
        # teleport models without holding the lock (service call)
        for i in range(self.n):
            if new_present[i] > 0.5 and old[i] < 0.5:
                self._teleport_to_centroid(i)
            elif new_present[i] < 0.5 and old[i] > 0.5:
                self._teleport_to_sentinel(i)

    # ---------------------------------------------------------- obs ----
    def _build_obs(self) -> dict:
        with self._lock:
            return {
                "pos":          self.poses_xy.tolist(),
                "vel":          self.vels_xy.tolist(),
                "goal":         [[0.0, GOAL_Y]] * self.n,
                "teleop_mask":  self.teleop_mask.tolist(),
                "present_mask": self.present_mask.tolist(),
                "time":         [[self.t]] * self.n,
            }

    # --------------------------------------------------------- tick ----
    def _tick(self) -> None:
        # Wait for the first /model_states packet so positions aren't zeros
        if not any(self.have_state):
            return

        obs = self._build_obs()
        with torch.no_grad():
            x = self.agent.format_input([obs], self.device)
            action, _, _, _ = self.agent.get_action_and_value(x)
        a = action[0].cpu().numpy()  # (n, 2) world-frame

        with self._lock:
            teleop_mask = self.teleop_mask.copy()
            present_mask = self.present_mask.copy()
            teleop_vels = self.teleop_vels.copy()

        # mirror env_hallway.vector_step: override teleop slots, zero non-present
        a = a * (1.0 - teleop_mask[:, None]) + teleop_vels * teleop_mask[:, None]
        a = a * present_mask[:, None]
        a = np.clip(a, -MAX_V, MAX_V)
        a = self._apply_soft_collision(a, present_mask)
        self._publish_action(a, present_mask)

        with self._lock:
            self.step_count += 1
            self.t += DT

        if self._check_done():
            if self.autoreset:
                self._reset_episode()
            else:
                self.get_logger().info("episode done, autoreset=false — not resetting")

    # --------------------------------------------------- publish ------
    def _publish_action(self, action_world: np.ndarray, present_mask: np.ndarray) -> None:
        with self._lock:
            yaws = self.yaws.copy()
        for i in range(self.n):
            if present_mask[i] < 0.5:
                continue
            vx_w, vy_w = float(action_world[i, 0]), float(action_world[i, 1])
            yaw = float(yaws[i])
            tw = Twist()
            # world -> body rotation. LIMO diff-drive will execute linear.x
            # and angular.z; linear.y is ignored (see plan risk #1).
            tw.linear.x = vx_w * math.cos(yaw) + vy_w * math.sin(yaw)
            tw.linear.y = -vx_w * math.sin(yaw) + vy_w * math.cos(yaw)
            # passive yaw damping toward yaw=0 keeps body~world aligned so the
            # ignored linear.y never has to do real work.
            tw.angular.z = -self.k_yaw * yaw
            self.cmd_pubs[i].publish(tw)

    def _apply_soft_collision(self, actions: np.ndarray, present_mask: np.ndarray) -> np.ndarray:
        """Push robots apart when closer than 2*AGENT_RADIUS — mirrors the ROS1
        reference. Belt-and-braces because diff-drive can lag and let two
        robots overlap before the policy reacts."""
        result = actions.copy()
        with self._lock:
            poses = self.poses_xy.copy()
        min_dist = 2.0 * AGENT_RADIUS
        for i in range(self.n):
            if present_mask[i] < 0.5:
                continue
            for j in range(i + 1, self.n):
                if present_mask[j] < 0.5:
                    continue
                delta = poses[i] - poses[j]
                dist = float(np.linalg.norm(delta))
                if 1e-6 < dist < min_dist:
                    repulsion = (delta / dist) * (min_dist - dist) * 5.0
                    result[i] += repulsion
                    result[j] -= repulsion
        return np.clip(result, -MAX_V, MAX_V)

    # ---------------------------------------------------- done/reset --
    def _check_done(self) -> bool:
        with self._lock:
            teleop = self.teleop_mask.copy()
            present = self.present_mask.copy()
            poses = self.poses_xy.copy()
            step_count = self.step_count
        active = np.where((present > 0.5) & (teleop < 0.5))[0]
        if len(active) > 0:
            centroid_y = float(poses[active, 1].mean())
            if centroid_y >= GOAL_Y and not self.goal_reached:
                self.goal_reached = True
                self.get_logger().info(
                    f"GOAL reached at step {step_count} (centroid_y={centroid_y:+.3f})"
                )
        timeout = step_count >= self.max_steps
        return self.goal_reached or timeout

    def _reset_episode(self) -> None:
        self.get_logger().info(
            f"episode reset (was step={self.step_count}, goal={self.goal_reached})"
        )
        # zero cmd_vel for all so the diff-drive plugin doesn't run on stale input
        zero = Twist()
        for p in self.cmd_pubs:
            p.publish(zero)

        # teleport: first num_agents to formation slots around (0, SPAWN_Y),
        # the rest (if they exist) to sentinel. Skip indices beyond
        # total_robots — those entities were never spawned in Gazebo and
        # calling SetEntityState on them spams the gzserver log.
        base = target_formation_positions(self.num_agents).cpu().numpy()
        rng = np.random.default_rng()
        for i in range(self.total_robots):
            if i < self.num_agents:
                jx, jy = rng.uniform(-0.05, 0.05, size=2)
                x = float(base[i, 0] + jx)
                y = float(SPAWN_Y + base[i, 1] + jy)
            else:
                x, y = float(SENTINEL_X), float(SENTINEL_Y)
            self._set_entity_pose(i, x, y, yaw=0.0)

        with self._lock:
            self.present_mask[:] = 0.0
            self.present_mask[: self.num_agents] = 1.0
            self.teleop_mask[:] = 0.0
            self.teleop_vels[:] = 0.0
            self._last_teleop_present = self.present_mask.copy()
            self.t = 0.0
            self.step_count = 0
            self.goal_reached = False
            self.episodes += 1

    # ----------------------------------------------- set_entity_state -
    def _teleport_to_centroid(self, robot_idx: int) -> None:
        """Mirrors env_hallway.spawn: place the new robot near the active centroid."""
        with self._lock:
            active = (self.present_mask > 0.5) & (self.teleop_mask < 0.5)
            active[robot_idx] = False  # exclude the one we're spawning
            if active.any():
                centroid = self.poses_xy[active].mean(axis=0)
            else:
                centroid = np.array([0.0, SPAWN_Y], dtype=np.float32)
        rng = np.random.default_rng()
        jitter = rng.uniform(-0.15, 0.15, size=2)
        self._set_entity_pose(
            robot_idx, float(centroid[0] + jitter[0]), float(centroid[1] + jitter[1]), 0.0
        )

    def _teleport_to_sentinel(self, robot_idx: int) -> None:
        self._set_entity_pose(robot_idx, float(SENTINEL_X), float(SENTINEL_Y), 0.0)

    def _set_entity_pose(self, robot_idx: int, x: float, y: float, yaw: float) -> None:
        if robot_idx >= self.total_robots:
            # Entity wasn't spawned; calling SetEntityState would log
            # "entity [limo_N] does not exist" every reset.
            return
        if not self.set_state.service_is_ready():
            return
        req = SetEntityState.Request()
        state = EntityState()
        state.name = f"{self.entity_prefix}{robot_idx + 1}"
        state.reference_frame = "world"
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0.0
        state.pose.orientation.w = math.cos(yaw / 2.0)
        state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state = state
        # async call — don't block the rclpy executor on the response
        self.set_state.call_async(req)

    # --------------------------------------------------- shutdown ------
    def stop_all_robots(self) -> None:
        """Publish a zero Twist to every cmd_vel topic so gazebo_ros_diff_drive
        doesn't keep executing the last random policy command after this node
        dies. Called from main()'s finally block on Ctrl+C."""
        zero = Twist()
        for pub in self.cmd_pubs:
            try:
                pub.publish(zero)
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = CircleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send zero Twist to every robot so gazebo_ros_diff_drive doesn't
        # keep replaying the last random command after circle_node dies.
        # Small sleep to let DDS deliver the messages before tearing the
        # node down.
        try:
            node.stop_all_robots()
            import time
            time.sleep(0.2)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
