"""FormationHallwayEnv — always-circle formation with dynamic robot count.

1..10 robots in an 8 x 8 m square arena. The active cluster (present and
non-teleop'd robots) always targets a circle centred on its own centroid;
radius scales with `n_active`. The cluster's job is to translate +y across
the arena from SPAWN_Y to GOAL_Y while holding the circle.

Buffers are sized to MAX_AGENTS=10. `present_mask` is the live source of
truth for "does this robot exist right now"; spawn() flips a slot to
present, delete() flips it back and parks the position at a far sentinel
so radius_graph drops it. `teleop_mask` continues to mean "this robot is
under human override"; teleop'd robots stay in the world but don't count
toward the circle and earn no policy reward.
"""
from __future__ import annotations

import math
import os
import sys
from collections import deque
from typing import List, Optional

import gymnasium as gym
import numpy as np
import pygame
import torch
from scipy.optimize import linear_sum_assignment

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contract import (
    AGENT_RADIUS,
    CIRCLE_SIDE,
    DEFAULT_MAX_TIME_STEPS,
    DEFAULT_RENDER_PX_PER_M,
    DT,
    GOAL_Y,
    INITIAL_AGENTS,
    MAX_A,
    MAX_AGENTS,
    MAX_V,
    MIN_A,
    MIN_AGENTS,
    REWARD_COEFFS,
    SENTINEL_X,
    SENTINEL_Y,
    SPAWN_Y,
    WORLD_H,
    WORLD_W,
)

X = 0
Y = 1


def target_formation_positions(n: int, scale: float = CIRCLE_SIDE) -> torch.Tensor:
    """Return `n` points on a circle centred at the origin.

    The circle is sized so adjacent neighbours sit `scale` apart along the
    chord; radius `r = scale / (2 sin(pi/n))`. Special cases:
      n=1 -> single point at origin.
      n=2 -> two points at (+/- scale/2, 0) (degenerate "circle" = line).
    """
    if n < 1:
        raise ValueError(f"target_formation_positions requires n >= 1, got {n}")
    if n == 1:
        return torch.zeros(1, 2, dtype=torch.float32)
    if n == 2:
        return torch.tensor(
            [[-scale / 2.0, 0.0], [scale / 2.0, 0.0]], dtype=torch.float32
        )
    r = scale / (2.0 * math.sin(math.pi / n))
    # First slot at angle pi/2 (i.e. straight up the +y axis) so n=4 lands
    # axis-aligned and the rendering is intuitive.
    angles = torch.tensor(
        [math.pi / 2.0 + 2.0 * math.pi * i / n for i in range(n)],
        dtype=torch.float32,
    )
    return torch.stack([r * torch.cos(angles), r * torch.sin(angles)], dim=-1)


class FormationHallwayEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config: Optional[dict] = None):
        cfg = dict(config or {})
        cfg.setdefault("n_agents", MAX_AGENTS)  # buffer size, not active count
        cfg.setdefault("initial_agents", INITIAL_AGENTS)
        cfg.setdefault("num_envs", 1)
        cfg.setdefault("dt", DT)
        cfg.setdefault("device", "cpu")
        cfg.setdefault("world_dim", (WORLD_W, WORLD_H))
        cfg.setdefault("max_v", MAX_V)
        cfg.setdefault("max_a", MAX_A)
        cfg.setdefault("min_a", MIN_A)
        cfg.setdefault("agent_radius", AGENT_RADIUS)
        cfg.setdefault("max_time_steps", DEFAULT_MAX_TIME_STEPS)
        cfg.setdefault("pos_noise_std", 0.0)
        cfg.setdefault("circle_side", CIRCLE_SIDE)
        cfg.setdefault("render_px_per_m", DEFAULT_RENDER_PX_PER_M)
        cfg.setdefault("spawn_y", SPAWN_Y)
        cfg.setdefault("goal_y", GOAL_Y)
        cfg.setdefault("reward_coeffs", dict(REWARD_COEFFS))
        cfg.setdefault("render", False)
        self.cfg = cfg

        n = self.cfg["n_agents"]
        if n != MAX_AGENTS:
            raise ValueError(
                f"n_agents must equal MAX_AGENTS={MAX_AGENTS}; got {n}. "
                "Dynamic count is implemented via present_mask, not by resizing buffers."
            )
        if not (MIN_AGENTS <= self.cfg["initial_agents"] <= MAX_AGENTS):
            raise ValueError(
                f"initial_agents must be in [{MIN_AGENTS}, {MAX_AGENTS}]; "
                f"got {self.cfg['initial_agents']}"
            )

        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Box(low=-cfg["max_v"], high=cfg["max_v"], shape=(2,), dtype=float),) * n
        )
        max_t = cfg["max_time_steps"] * cfg["dt"]
        self.observation_space = gym.spaces.Dict(
            {
                "pos": gym.spaces.Box(-WORLD_H, WORLD_H, shape=(n, 2), dtype=float),
                "vel": gym.spaces.Box(-1e5, 1e5, shape=(n, 2), dtype=float),
                "goal": gym.spaces.Box(-WORLD_H, WORLD_H, shape=(n, 2), dtype=float),
                "teleop_mask": gym.spaces.Box(0.0, 1.0, shape=(n,), dtype=float),
                "present_mask": gym.spaces.Box(0.0, 1.0, shape=(n,), dtype=float),
                "time": gym.spaces.Box(0.0, max_t, shape=(n, 1), dtype=float),
            }
        )

        self.device = torch.device(cfg["device"])
        self.vec_p_shape = (cfg["num_envs"], n, 2)

        # stall detection: ring buffer of cluster centroids per env
        self._stall_window = int(cfg["reward_coeffs"]["stall_window"])
        self._centroid_history: List[deque] = [
            deque(maxlen=self._stall_window) for _ in range(cfg["num_envs"])
        ]

        self.vector_reset()

        self.display = None
        if cfg.get("render", False):
            pygame.init()
            size = (
                int(cfg["world_dim"][0] * cfg["render_px_per_m"]),
                int(cfg["world_dim"][1] * cfg["render_px_per_m"]),
            )
            self.display = pygame.display.set_mode(size)

    # --- utilities -------------------------------------------------------
    def create_state_tensor(self) -> torch.Tensor:
        return torch.zeros(self.vec_p_shape, dtype=torch.float32, device=self.device)

    def sample_pos_noise(self) -> torch.Tensor:
        std = self.cfg["pos_noise_std"]
        if std > 0.0:
            return torch.normal(0.0, std, self.vec_p_shape, device=self.device)
        return self.create_state_tensor()

    def compute_agent_dists(self, ps: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(ps, ps)
        n = ps.shape[1]
        diag = torch.eye(n, device=ps.device).bool().unsqueeze(0).expand(ps.shape[0], -1, -1)
        d = d.masked_fill(diag, float("inf"))
        return d

    def rand(self, size, a: float, b: float) -> torch.Tensor:
        return (b - a) * torch.rand(size, device=self.device) + a

    # --- formation helper -----------------------------------------------
    def target_formation_positions(self, n: int) -> torch.Tensor:
        return target_formation_positions(n, self.cfg["circle_side"])

    # --- spawn / reset ---------------------------------------------------
    def _sentinel_pos(self) -> torch.Tensor:
        return torch.tensor([SENTINEL_X, SENTINEL_Y], dtype=torch.float32, device=self.device)

    def _initial_positions(self, num_envs: int, initial_n: int):
        """Place `initial_n` robots near (0, SPAWN_Y) on a small jittered circle.

        Slots [0, initial_n) are present; the rest are parked at the sentinel.
        """
        n = self.cfg["n_agents"]
        sentinel = self._sentinel_pos()
        starts = sentinel.view(1, 1, 2).expand(num_envs, n, 2).clone()
        if initial_n > 0:
            base = self.target_formation_positions(initial_n)  # (initial_n, 2)
            starts[:, :initial_n, :] = base.unsqueeze(0).repeat(num_envs, 1, 1).to(self.device)
            starts[:, :initial_n, X] += self.rand((num_envs, initial_n), -0.05, 0.05)
            starts[:, :initial_n, Y] += self.rand((num_envs, initial_n), -0.05, 0.05) + self.cfg["spawn_y"]
        goals = torch.zeros(num_envs, n, 2, device=self.device)
        goals[:, :, Y] = self.cfg["goal_y"]
        return starts, goals

    def vector_reset(self):
        n = self.cfg["n_agents"]
        nE = self.cfg["num_envs"]
        initial_n = int(self.cfg["initial_agents"])
        starts, goals = self._initial_positions(nE, initial_n)
        self.ps = starts
        self.goal_ps = goals
        self.measured_vs = self.create_state_tensor()
        self.teleop_mask = torch.zeros(nE, n, dtype=torch.float32, device=self.device)
        self.present_mask = torch.zeros(nE, n, dtype=torch.float32, device=self.device)
        self.present_mask[:, :initial_n] = 1.0
        self.teleop_vels = self.create_state_tensor()
        self.timesteps = torch.zeros(nE, dtype=torch.int32, device=self.device)
        self.goal_reached = torch.zeros(nE, dtype=torch.bool, device=self.device)
        for h in self._centroid_history:
            h.clear()
        return [self.get_obs(i) for i in range(nE)]

    def reset_at(self, index: int):
        initial_n = int(self.cfg["initial_agents"])
        start, goal = self._initial_positions(1, initial_n)
        n = self.cfg["n_agents"]
        self.ps[index] = start[0]
        self.goal_ps[index] = goal[0]
        self.measured_vs[index] = 0.0
        self.teleop_mask[index] = 0.0
        self.present_mask[index] = 0.0
        self.present_mask[index, :initial_n] = 1.0
        self.teleop_vels[index] = 0.0
        self.timesteps[index] = 0
        self.goal_reached[index] = False
        self._centroid_history[index].clear()
        return self.get_obs(index)

    # --- teleop interface (used by training disturbance + keyboard) -----
    def set_teleop(self, env_idx: int, robot_idx: int, active: bool):
        if active and float(self.present_mask[env_idx, robot_idx]) < 0.5:
            return  # can't teleop a non-present robot
        self.teleop_mask[env_idx, robot_idx] = 1.0 if active else 0.0
        if not active:
            self.teleop_vels[env_idx, robot_idx] = 0.0

    def set_teleop_action(self, env_idx: int, robot_idx: int, vel):
        v = torch.as_tensor(vel, dtype=torch.float32, device=self.device)
        v = torch.clamp(v, -self.cfg["max_v"], self.cfg["max_v"])
        self.teleop_vels[env_idx, robot_idx] = v

    # --- spawn / delete API ---------------------------------------------
    def n_present(self, env_idx: int) -> int:
        return int(self.present_mask[env_idx].sum().item())

    def spawn(self, env_idx: int, robot_idx: Optional[int] = None) -> Optional[int]:
        """Bring a robot online. Returns the slot used, or None on no-op.

        If `robot_idx` is None, picks the lowest free index. Spawn position is
        the active-cluster centroid plus small jitter (or SPAWN_Y if no robots
        are present yet). Velocity zero, teleop cleared.
        """
        n = self.cfg["n_agents"]
        if robot_idx is None:
            free = (self.present_mask[env_idx] < 0.5).nonzero(as_tuple=True)[0]
            if free.numel() == 0:
                return None
            robot_idx = int(free[0].item())
        if not (0 <= robot_idx < n):
            raise IndexError(f"robot_idx {robot_idx} out of [0, {n})")
        if float(self.present_mask[env_idx, robot_idx]) > 0.5:
            return None
        if self.n_present(env_idx) >= MAX_AGENTS:
            return None

        # Spawn near active centroid (if any), else near SPAWN_Y on the y-axis
        active = (self.present_mask[env_idx] > 0.5) & (self.teleop_mask[env_idx] < 0.5)
        present = self.present_mask[env_idx] > 0.5
        if active.any():
            centroid = self.ps[env_idx][active].mean(dim=0)
        elif present.any():
            centroid = self.ps[env_idx][present].mean(dim=0)
        else:
            centroid = torch.tensor(
                [0.0, self.cfg["spawn_y"]], dtype=torch.float32, device=self.device
            )
        jitter = (torch.rand(2, device=self.device) - 0.5) * 0.1
        self.ps[env_idx, robot_idx] = centroid + jitter
        self.measured_vs[env_idx, robot_idx] = 0.0
        self.present_mask[env_idx, robot_idx] = 1.0
        self.teleop_mask[env_idx, robot_idx] = 0.0
        self.teleop_vels[env_idx, robot_idx] = 0.0
        return robot_idx

    def delete(self, env_idx: int, robot_idx: int) -> bool:
        """Take a robot offline. Returns True if the slot was actually freed.

        No-op if robot is not present, or if it's the last present robot
        (preserves the n_present >= MIN_AGENTS=1 invariant).
        """
        if not (0 <= robot_idx < self.cfg["n_agents"]):
            raise IndexError(f"robot_idx {robot_idx} out of [0, {self.cfg['n_agents']})")
        if float(self.present_mask[env_idx, robot_idx]) < 0.5:
            return False
        if self.n_present(env_idx) <= MIN_AGENTS:
            return False
        self.present_mask[env_idx, robot_idx] = 0.0
        self.teleop_mask[env_idx, robot_idx] = 0.0
        self.teleop_vels[env_idx, robot_idx] = 0.0
        self.measured_vs[env_idx, robot_idx] = 0.0
        self.ps[env_idx, robot_idx] = self._sentinel_pos()
        return True

    # --- obs -------------------------------------------------------------
    def get_obs(self, index: int):
        n = self.cfg["n_agents"]
        t = (self.timesteps[index] * self.cfg["dt"]).item()
        return {
            "pos": self.ps[index].tolist(),
            "vel": self.measured_vs[index].tolist(),
            "goal": self.goal_ps[index].tolist(),
            "teleop_mask": self.teleop_mask[index].tolist(),
            "present_mask": self.present_mask[index].tolist(),
            "time": [[t]] * n,
        }

    # --- reward components ----------------------------------------------
    def _formation_reward(self, env_idx: int, ps: torch.Tensor):
        """Per-robot circle-formation penalty for env_idx.

        Hungarian-assigns active robots (present & non-teleop'd) to slots of
        the target circle centred at the active centroid. Returns
        (per_robot_penalty (n_agents,), mean_slot_distance or None).
        """
        n = self.cfg["n_agents"]
        out = torch.zeros(n, device=ps.device)
        active_mask = (self.present_mask[env_idx] > 0.5) & (self.teleop_mask[env_idx] < 0.5)
        active_idx = active_mask.nonzero(as_tuple=True)[0]
        k = active_idx.numel()
        if k < 2:
            return out, None
        active_ps = ps[active_idx]  # (k, 2)
        centroid = active_ps.mean(dim=0)
        slots = self.target_formation_positions(k).to(ps.device) + centroid  # (k, 2)
        cost = torch.cdist(active_ps, slots).cpu().numpy()
        row, col = linear_sum_assignment(cost)
        coeff = self.cfg["reward_coeffs"]["k_form"]
        dists = []
        for ri, ci in zip(row, col):
            robot = int(active_idx[ri].item())
            d = float(cost[ri, ci])
            dists.append(d)
            out[robot] = -coeff * d
        return out, sum(dists) / len(dists)

    def _stall_penalty(self, env_idx: int, centroid: torch.Tensor) -> float:
        h = self._centroid_history[env_idx]
        h.append(centroid.detach().clone())
        if len(h) < self._stall_window:
            return 0.0
        moved = float(torch.linalg.norm(h[-1] - h[0]).item())
        if moved < self.cfg["reward_coeffs"]["stall_eps"]:
            return -float(self.cfg["reward_coeffs"]["k_stall"])
        return 0.0

    # --- step -----------------------------------------------------------
    def vector_step(self, actions):
        cfg = self.cfg
        n = cfg["n_agents"]
        nE = cfg["num_envs"]
        coeffs = cfg["reward_coeffs"]

        actions_t = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
        # Override teleop slots with stored teleop velocities; non-present
        # slots get zero (their positions are anchored at the sentinel anyway).
        teleop3 = self.teleop_mask.unsqueeze(-1)
        present3 = self.present_mask.unsqueeze(-1)
        actions_t = actions_t * (1.0 - teleop3) + self.teleop_vels * teleop3
        actions_t = actions_t * present3  # zero out non-present

        desired_vs = torch.clip(actions_t, -cfg["max_v"], cfg["max_v"])
        desired_as = (desired_vs - self.measured_vs) / cfg["dt"]
        possible_as = torch.clip(desired_as, cfg["min_a"], cfg["max_a"])
        possible_vs = self.measured_vs + possible_as * cfg["dt"]

        previous_ps = self.ps.clone()
        rewards = torch.zeros(nE, n, device=self.device)

        # Per-agent collision check among PRESENT robots — non-present robots
        # sit at the sentinel and are far enough away to never collide, but we
        # still skip them explicitly so a sentinel-coincident edge case can't bite.
        next_ps = self.ps.clone()
        present_bool = self.present_mask > 0.5
        for i in range(n):
            present_i = present_bool[:, i]
            if not present_i.any():
                continue
            trial = next_ps.clone()
            trial[:, i] += possible_vs[:, i] * cfg["dt"]
            d = self.compute_agent_dists(trial)[:, i]  # (nE, n) infs on diag
            # Mask distances against non-present neighbours (set to inf)
            non_present = ~present_bool
            d = d.masked_fill(non_present, float("inf"))
            collide = (torch.min(d, dim=1)[0] <= 2 * cfg["agent_radius"]) & present_i
            update = present_i & ~collide
            next_ps[update, i] = trial[update, i]
            rewards[collide, i] -= coeffs["k_coll"]

        # Wall containment in x (only matters for present robots; sentinel is far)
        half_w = cfg["world_dim"][0] / 2.0 - cfg["agent_radius"]
        overshoot_x = (next_ps[:, :, X].abs() - half_w).clamp(min=0.0) * self.present_mask
        rewards -= coeffs["k_wall"] * overshoot_x
        # Clamp x for present robots only — leave sentinel positions alone
        clamped_x = torch.clip(next_ps[:, :, X], -half_w, half_w)
        next_ps[:, :, X] = torch.where(present_bool, clamped_x, next_ps[:, :, X])

        # y soft bounds — only for present robots
        half_h = cfg["world_dim"][1] / 2.0
        clamped_y = torch.clip(next_ps[:, :, Y], -half_h, half_h)
        next_ps[:, :, Y] = torch.where(present_bool, clamped_y, next_ps[:, :, Y])

        # Position noise applied only to present robots (don't perturb sentinel)
        noise = self.sample_pos_noise()
        noise = noise * self.present_mask.unsqueeze(-1)
        next_ps = next_ps + noise
        self.ps = next_ps
        self.measured_vs = (self.ps - previous_ps) / cfg["dt"]
        # Force measured_vs of non-present to zero so loss / metrics aren't poisoned
        self.measured_vs = self.measured_vs * self.present_mask.unsqueeze(-1)

        # Forward progress (only credited to active robots)
        dy = self.ps[:, :, Y] - previous_ps[:, :, Y]
        active = self.present_mask * (1.0 - self.teleop_mask)  # (nE, n)
        rewards += coeffs["k_fwd"] * dy * active

        # Formation + stall + goal — per env
        formation_errs: List[Optional[float]] = [None] * nE
        stalled_flags = [False] * nE
        for e in range(nE):
            penalty, err = self._formation_reward(e, self.ps[e])
            rewards[e] += penalty * active[e]
            formation_errs[e] = err

            active_e = active[e].bool()
            if active_e.any():
                centroid = self.ps[e][active_e].mean(dim=0)
                stall_pen = self._stall_penalty(e, centroid)
                if stall_pen != 0.0:
                    rewards[e] += stall_pen * active[e]
                    stalled_flags[e] = True

                # Goal bonus: cluster centroid past GOAL_Y (one-shot)
                if not bool(self.goal_reached[e]) and float(centroid[Y].item()) >= cfg["goal_y"]:
                    rewards[e] += coeffs["k_goal"] * active[e]
                    self.goal_reached[e] = True

        # Zero reward for teleop'd / non-present robots
        rewards = rewards * active

        self.timesteps += 1
        timeout = self.timesteps >= cfg["max_time_steps"]
        empty = self.present_mask.sum(dim=1) <= 0  # all-deleted (degenerate)
        dones = (timeout | self.goal_reached | empty).tolist()
        obs = [self.get_obs(i) for i in range(nE)]
        infos = []
        for e in range(nE):
            active_e = active[e].bool()
            mean_dy = (
                float((dy[e][active_e]).mean().item()) / cfg["dt"] if active_e.any() else 0.0
            )
            wall_hit = bool((overshoot_x[e] > 0).any().item())
            collided = bool((rewards[e] <= -coeffs["k_coll"] + 1e-6).any().item())
            n_pres = int(self.present_mask[e].sum().item())
            n_act = int(active[e].sum().item())
            infos.append(
                {
                    "rewards": {k: float(rewards[e, k].item()) for k in range(n)},
                    "active_count": n_act,
                    "n_present": n_pres,
                    "n_active": n_act,
                    "n_teleop": int(self.teleop_mask[e].sum().item()),
                    "circle_radius": (
                        float(CIRCLE_SIDE / (2.0 * math.sin(math.pi / max(n_act, 2))))
                        if n_act >= 2 else 0.0
                    ),
                    "goal_reached": bool(self.goal_reached[e].item()),
                    "formation_error": formation_errs[e],
                    "fwd_velocity": mean_dy,
                    "stalled": stalled_flags[e],
                    "wall_hit": wall_hit,
                    "collided": collided,
                }
            )
        return obs, torch.sum(rewards, dim=1).tolist(), dones, infos

    def get_unwrapped(self):
        return []

    def close(self):
        if self.display is not None:
            pygame.display.quit()
            pygame.quit()
            self.display = None


# Lightweight single-env render wrapper — mirrors PassageEnvRender from env_line
class FormationHallwayEnvRender(FormationHallwayEnv):
    def __init__(self, config: Optional[dict] = None):
        config = dict(config or {})
        config["num_envs"] = 1
        config.setdefault("render", True)
        super().__init__(config)

    def reset(self):
        return self.reset_at(0)

    def step(self, actions):
        a = np.zeros((1, self.cfg["n_agents"], 2), dtype=np.float32)
        a[0] = np.asarray(actions, dtype=np.float32)
        obs, r, done, info = self.vector_step(a)
        return obs[0], r[0], done[0], info[0]


def _smoke_test(steps: int = 200, seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = FormationHallwayEnv({"num_envs": 4, "initial_agents": 4})
    env.vector_reset()
    rng = np.random.default_rng(seed)
    total_r = np.zeros(env.cfg["num_envs"])
    for t in range(steps):
        a = rng.uniform(-MAX_V, MAX_V, size=(env.cfg["num_envs"], MAX_AGENTS, 2))
        obs, r, done, info = env.vector_step(a)
        total_r += np.array(r)
        if t == 50:
            env.set_teleop(0, 1, True)
            env.set_teleop_action(0, 1, np.array([0.5, 0.0]))
        if t == 100:
            env.spawn(0)
            env.spawn(0)
        if t == 150:
            env.delete(0, 0)
        if any(done):
            for i, d in enumerate(done):
                if d:
                    env.reset_at(i)
    print(f"smoke OK: steps={steps}, mean total reward={total_r.mean():.3f}")
    print(f"final present_mask[0]={env.present_mask[0].tolist()}")
    print(f"final teleop_mask[0]={env.teleop_mask[0].tolist()}")
    print(f"n_present[0]={env.n_present(0)}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--random", action="store_true")
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()
    _smoke_test(args.steps)
