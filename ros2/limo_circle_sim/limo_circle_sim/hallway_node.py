#!/usr/bin/env python3
"""
ROS2 wrapper around the policy_v2 (older "hallway") PPO checkpoint.

Mirrors circle_node.py structurally. Differences are confined to:

- MAX_AGENTS = 4              (v2's contract.MAX_AGENTS, not v1's 10)
- comm_range = 2.0            (v2's training value, not v1's 4.0)
- target_formation_positions  (square/triangle/line — NOT a circle)
- AGENT_RADIUS = 0.08          (v2's contract, smaller cluster)
- FORMATION_SCALE = 0.35       (compact 0.35 m side length)

The 8x8 m circle_arena world is reused as-is — policy_v2 was trained on a
2x12 hallway but reads its goal direction from obs.goal at runtime, so
feeding goal=[0, +3.5] (matching circle_node's deployment goal) makes the
cluster march toward the green line in our arena. Spawn point at
(0, spawn_y_default=-3.2) is the same as circle_node's spawn.

Everything below is intentionally self-contained for v2 semantics. The
only imports from code/ are the model architecture (same GNN code as v1)
and the checkpoint loader (format identical between v1 and v2).

Test sequence the user runs against this node:
    spawn 4 (square) → delete to 2 (line) → spawn back to 4 (square)
all live, via teleop_node's +/- keys.
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
from gazebo_msgs.msg import ModelStates, EntityState
from gazebo_msgs.srv import SetEntityState

# ----------------------------------------------------------------- code/ shim
# Reuse model.py + checkpoint.py from the repo's code/ — they are
# architecture-identical to policy_v2 (GNN with Beta policy head). We
# specifically do NOT import contract.py or env_hallway.py from code/
# because those carry v1's circle constants (MAX_AGENTS=10, n-gon formation).
_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[3]
_CODE = _REPO / "code"
if not _CODE.is_dir():
    raise RuntimeError(f"could not find policy code at {_CODE}")
sys.path.insert(0, str(_CODE))

import torch  # noqa: E402  (after sys.path tweak)
from model import Agent  # noqa: E402
from checkpoint import load_checkpoint  # noqa: E402


# ------------------------------------------------------ policy_v2 contract --
# These are duplicated locally instead of imported so this node never picks
# up v1's contract.py from `code/` on this branch. Values copied verbatim
# from upstream/aneesh/policy_v2:code/contract.py
MAX_AGENTS = 4
DT = 0.05
MAX_V = 1.0
AGENT_RADIUS = 0.08
FORMATION_SCALE = 0.35
# WORLD_W / WORLD_H from v2's contract are (2.0, 12.0). We only use these
# as observation-space bounds; the policy reads the live obs.pos so it
# doesn't actually constrain runtime behavior. Use generous bounds.
OBS_BOUND = 12.0
# Sentinel position for non-present robots — matches circle_node's choice.
SENTINEL_X = 24.0
SENTINEL_Y = 24.0


def target_formation_positions_v2(n: int, scale: float = FORMATION_SCALE) -> torch.Tensor:
    """Canonical formation slots, centred at origin. Verbatim from v2's
    env_hallway.target_formation_positions (lines 46-72 on policy_v2).

      n=4 → square (vertices at ±s/2)
      n=3 → equilateral triangle, one vertex pointing +y
      n=2 → horizontal line, separation s
      n=1 → single point at origin
    """
    s = scale
    if n == 4:
        return torch.tensor(
            [[-s / 2, -s / 2], [s / 2, -s / 2], [s / 2, s / 2], [-s / 2, s / 2]],
            dtype=torch.float32,
        )
    if n == 3:
        h = s / (2 * 3 ** 0.5)
        H = s / (3 ** 0.5)
        return torch.tensor(
            [[-s / 2, -h], [s / 2, -h], [0.0, H]],
            dtype=torch.float32,
        )
    if n == 2:
        return torch.tensor([[-s / 2, 0.0], [s / 2, 0.0]], dtype=torch.float32)
    if n == 1:
        return torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    raise ValueError(f"unsupported active_count={n}")


# ------------------------------------------------------------------ math --
def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# -------------------------------------------------------------- env shim --
class HallwayEnvShim:
    """Quacks like v2's FormationHallwayEnv for Agent.__init__. No physics
    — the only members touched by Agent are observation_space, action_space
    and cfg. Shape locked to MAX_AGENTS=4."""

    def __init__(self):
        import gymnasium as gym
        n = MAX_AGENTS
        self.cfg = {"n_agents": n, "num_envs": 1, "dt": DT, "max_v": MAX_V}
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Box(low=-MAX_V, high=MAX_V, shape=(2,), dtype=float),) * n
        )
        max_t = 3000 * DT
        self.observation_space = gym.spaces.Dict({
            "pos":          gym.spaces.Box(-OBS_BOUND, OBS_BOUND, shape=(n, 2), dtype=float),
            "vel":          gym.spaces.Box(-1e5, 1e5,             shape=(n, 2), dtype=float),
            "goal":         gym.spaces.Box(-OBS_BOUND, OBS_BOUND, shape=(n, 2), dtype=float),
            "teleop_mask":  gym.spaces.Box(0.0, 1.0,              shape=(n,),   dtype=float),
            "present_mask": gym.spaces.Box(0.0, 1.0,              shape=(n,),   dtype=float),
            "time":         gym.spaces.Box(0.0, max_t,            shape=(n, 1), dtype=float),
        })


# ----------------------------------------------------------------- node --
class HallwayNode(Node):
    def __init__(self) -> None:
        super().__init__("hallway_node")

        # ---- params -----------------------------------------------------
        self.declare_parameter("weights", "")
        self.declare_parameter("num_agents", 4)
        self.declare_parameter("max_agents", MAX_AGENTS)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("k_yaw", 0.0)
        self.declare_parameter("max_steps", 3000)
        self.declare_parameter("autoreset", True)
        self.declare_parameter("entity_prefix", "limo_")
        self.declare_parameter("total_robots", 0)
        # spawn_y default -3.2 matches circle_sim.launch.py's SPAWN_Y_LAUNCH.
        # spawn_yaw default π/2 puts body-+X along world-+Y so the diff-drive
        # forward axis aligns with goal direction (irrelevant with planar_move
        # plugin but kept consistent with circle_node's setup).
        self.declare_parameter("spawn_y", -3.2)
        self.declare_parameter("spawn_yaw", math.pi / 2.0)
        # goal_y is what we feed into obs.goal[*,1]. Default 3.5 to match
        # the green goal line in circle_arena.world. The policy was trained
        # with GOAL_Y=5.0 (12 m hallway), but it reads obs.goal at runtime
        # so we can retarget it without retraining.
        self.declare_parameter("goal_y", 3.5)

        weights = str(self.get_parameter("weights").value)
        if not weights or not os.path.isfile(weights):
            raise RuntimeError(
                f"param `weights` must point at a .pt checkpoint; got {weights!r}"
            )
        self.num_agents = int(self.get_parameter("num_agents").value)
        self.max_agents = int(self.get_parameter("max_agents").value)
        if self.max_agents != MAX_AGENTS:
            raise RuntimeError(
                f"max_agents={self.max_agents} != policy_v2.MAX_AGENTS={MAX_AGENTS}; "
                "the policy network is sized for 4 — refusing to load."
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
        tr = int(self.get_parameter("total_robots").value)
        self.total_robots = tr if tr > 0 else self.num_agents
        if not (self.num_agents <= self.total_robots <= self.max_agents):
            raise RuntimeError(
                f"total_robots={self.total_robots} must be in "
                f"[num_agents={self.num_agents}, max_agents={self.max_agents}]"
            )
        self.spawn_y = float(self.get_parameter("spawn_y").value)
        self.spawn_yaw = float(self.get_parameter("spawn_yaw").value)
        self.goal_y = float(self.get_parameter("goal_y").value)

        # ---- load policy ------------------------------------------------
        env_shim = HallwayEnvShim()
        agent_cfg = {
            "model": {
                "custom_model_config": {
                    "activation": "relu",
                    "msg_features": 32,
                    "comm_range": 2.0,         # ← v2's training value
                    "use_masks": True,
                }
            }
        }
        self.agent = Agent(env_shim, agent_cfg).to(self.device)
        ckpt = load_checkpoint(weights, self.device)
        # strict=False is critical here: this code/ tree is on v1, the
        # checkpoint is v2-trained. Architectures are identical but the
        # state_dict's parameter names should map 1:1; if any don't, we
        # log and continue rather than refusing to load.
        missing, unexpected = self.agent.load_state_dict(
            ckpt["agent"], strict=False,
        )
        if missing or unexpected:
            self.get_logger().warn(
                f"checkpoint partial load: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys (v1<->v2 arch drift)"
            )
        self.agent.eval()
        self.get_logger().info(
            f"loaded weights={weights} (iter={ckpt.get('iteration', 0)})"
        )

        # ---- state ------------------------------------------------------
        self.n = MAX_AGENTS
        self._lock = threading.Lock()

        self.poses_xy = np.zeros((self.n, 2), dtype=np.float32)
        self.vels_xy = np.zeros((self.n, 2), dtype=np.float32)
        self.yaws = np.zeros(self.n, dtype=np.float32)
        self.have_state = [False] * self.n

        self.present_mask = np.zeros(self.n, dtype=np.float32)
        self.present_mask[: self.num_agents] = 1.0
        self.teleop_mask = np.zeros(self.n, dtype=np.float32)
        self.teleop_vels = np.zeros((self.n, 2), dtype=np.float32)
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

        self.set_state = self.create_client(SetEntityState, "/set_entity_state")
        if not self.set_state.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn(
                "/set_entity_state not available — episode reset and "
                "spawn/delete teleports will fail. Did the world load "
                "with gazebo_ros_state?"
            )

        self._reset_episode()
        self.timer = self.create_timer(DT, self._tick)
        self.get_logger().info(
            f"hallway_node up: max_agents={self.n} initial_n={self.num_agents} "
            f"dt={DT}s ({int(round(1.0 / DT))} Hz) goal_y={self.goal_y}"
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
        if len(msg.data) != self.n:
            return
        new_present = np.asarray(msg.data, dtype=np.float32)
        with self._lock:
            old = self._last_teleop_present.copy()
            self._last_teleop_present = new_present.copy()
            self.present_mask = new_present.copy()
            became_absent = (old > 0.5) & (new_present < 0.5)
            self.teleop_mask[became_absent] = 0.0
            self.teleop_vels[became_absent] = 0.0
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
                # NOTE: goal_y is the deployment goal (default 3.5), not v2's
                # training GOAL_Y=5.0. Policy reads this at every tick, so
                # retargeting via the param is enough.
                "goal":         [[0.0, self.goal_y]] * self.n,
                "teleop_mask":  self.teleop_mask.tolist(),
                "present_mask": self.present_mask.tolist(),
                "time":         [[self.t]] * self.n,
            }

    # --------------------------------------------------------- tick ----
    def _tick(self) -> None:
        if not any(self.have_state):
            return

        obs = self._build_obs()
        with torch.no_grad():
            x = self.agent.format_input([obs], self.device)
            action, _, _, _ = self.agent.get_action_and_value(x)
        a = action[0].cpu().numpy()

        with self._lock:
            teleop_mask = self.teleop_mask.copy()
            present_mask = self.present_mask.copy()
            teleop_vels = self.teleop_vels.copy()

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
        """World→body rotation for the planar_move plugin. Identical math
        to circle_node._publish_action — the plugin's body→world rotation
        cancels this, so (vx_w, vy_w) actually moves the body in those
        world directions regardless of yaw."""
        with self._lock:
            yaws = self.yaws.copy()

        for i in range(self.n):
            if present_mask[i] < 0.5:
                continue
            vx_w = float(action_world[i, 0])
            vy_w = float(action_world[i, 1])
            yaw = float(yaws[i])
            tw = Twist()
            tw.linear.x = vx_w * math.cos(yaw) + vy_w * math.sin(yaw)
            tw.linear.y = -vx_w * math.sin(yaw) + vy_w * math.cos(yaw)
            tw.angular.z = -self.k_yaw * yaw
            self.cmd_pubs[i].publish(tw)

    def _apply_soft_collision(self, actions: np.ndarray, present_mask: np.ndarray) -> np.ndarray:
        """Same as circle_node's, but with v2's AGENT_RADIUS=0.08 — the
        formation is much tighter so the collision push needs a smaller
        minimum-distance threshold."""
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
        """Mirrors v2's env_hallway goal-bonus check: cluster centroid past
        goal_y (one-shot). Active subset excludes teleop'd robots so a human
        holding one back doesn't keep an episode going forever."""
        with self._lock:
            teleop = self.teleop_mask.copy()
            present = self.present_mask.copy()
            poses = self.poses_xy.copy()
            step_count = self.step_count
        active = np.where((present > 0.5) & (teleop < 0.5))[0]
        if len(active) > 0:
            centroid_y = float(poses[active, 1].mean())
            if centroid_y >= self.goal_y and not self.goal_reached:
                self.goal_reached = True
                self.get_logger().info(
                    f"GOAL reached at step {step_count} "
                    f"(centroid_y={centroid_y:+.3f} ≥ goal_y={self.goal_y})"
                )
        timeout = step_count >= self.max_steps
        return self.goal_reached or timeout

    def _reset_episode(self) -> None:
        self.get_logger().info(
            f"episode reset (was step={self.step_count}, goal={self.goal_reached})"
        )
        zero = Twist()
        for p in self.cmd_pubs:
            p.publish(zero)

        # Teleport active robots into the v2-style formation (square,
        # triangle, line, or point) centered at (0, spawn_y). The shape
        # changes with num_agents — _reset_episode is called both at
        # startup and after a goal-reached, so the formation matches
        # whatever active count we're at right now.
        base = target_formation_positions_v2(self.num_agents).cpu().numpy()
        rng = np.random.default_rng()
        for i in range(self.total_robots):
            if i < self.num_agents:
                jx, jy = rng.uniform(-0.02, 0.02, size=2)  # tighter jitter than v1
                x = float(base[i, 0] + jx)
                y = float(self.spawn_y + base[i, 1] + jy)
                yaw = self.spawn_yaw
            else:
                x, y = float(SENTINEL_X), float(SENTINEL_Y)
                yaw = 0.0
            self._set_entity_pose(i, x, y, yaw=yaw)

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
        with self._lock:
            active = (self.present_mask > 0.5) & (self.teleop_mask < 0.5)
            active[robot_idx] = False
            if active.any():
                centroid = self.poses_xy[active].mean(axis=0)
            else:
                centroid = np.array([0.0, self.spawn_y], dtype=np.float32)
        rng = np.random.default_rng()
        jitter = rng.uniform(-0.10, 0.10, size=2)
        self._set_entity_pose(
            robot_idx, float(centroid[0] + jitter[0]),
            float(centroid[1] + jitter[1]), self.spawn_yaw,
        )

    def _teleport_to_sentinel(self, robot_idx: int) -> None:
        self._set_entity_pose(robot_idx, float(SENTINEL_X), float(SENTINEL_Y), 0.0)

    def _set_entity_pose(self, robot_idx: int, x: float, y: float, yaw: float) -> None:
        if robot_idx >= self.total_robots:
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
        self.set_state.call_async(req)

    # --------------------------------------------------- shutdown ------
    def stop_all_robots(self) -> None:
        import time
        zero = Twist()
        for _ in range(10):
            for pub in self.cmd_pubs:
                try:
                    pub.publish(zero)
                except Exception:
                    pass
            time.sleep(0.05)


def main(args=None):
    rclpy.init(args=args)
    node = HallwayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop_all_robots()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
