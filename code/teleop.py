"""Teleop drivers for FormationHallwayEnv.

RandomTeleop  — synthetic disturbance for training. Supports multiple
concurrent grabs (so the policy sees 1-, 2-, 3-, and 4-active regimes)
and an initial-regime distribution that pre-marks robots teleop'd at
episode reset (so every active count is well-represented from t=0).

KeyboardTeleop — pygame key handler for the eval/demo binary.

Both call env.set_teleop(env_idx, robot_idx, active) and
env.set_teleop_action(env_idx, robot_idx, vel) — see code/contract.py.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contract import MAX_AGENTS, MAX_V


@dataclass
class _Grab:
    robot: int
    age: int
    duration: int
    drift_dir: float   # +1 / -1 lateral push (0 -> hold still)
    base_vy: float
    drift_speed: float


# default mass on each active_count in {1,2,3,4} at episode start.
# slightly over-weights the small-cluster regimes vs a uniform 0.25/each
# because they were under-represented in earlier training runs.
DEFAULT_INIT_REGIME_DIST = [0.10, 0.25, 0.30, 0.35]


class RandomTeleop:
    """Per-env synthetic teleop disturbance.

    Two complementary mechanisms keep all active counts represented in
    training:

    * Multi-grab: at any time up to ``max_concurrent_grabs`` robots in an
      env can be teleop'd. Each grab is independent and runs the same
      sinusoidal lateral push as before.
    * Initial regime: at ``reset_env``, the active count is sampled from
      ``init_regime_dist`` and the matching number of robots are pre-marked
      teleop'd (held still at their spawn position). They release after a
      randomized opening duration like any other grab.
    """

    def __init__(
        self,
        env,
        p_grab: float = 0.005,
        p_release: float = 0.01,
        drift_speed: float = 0.6,
        min_duration: int = 40,
        max_duration: int = 160,
        max_concurrent_grabs: int = 3,
        init_regime_dist: Optional[List[float]] = None,
        init_hold_min: int = 30,
        init_hold_max: int = 80,
        seed: int = 0,
    ):
        self.env = env
        self.num_envs = env.cfg["num_envs"]
        self.n_agents = env.cfg["n_agents"]
        self.p_grab = p_grab
        self.p_release = p_release
        self.drift_speed = drift_speed
        self.min_duration = min_duration
        self.max_duration = max_duration
        # Cap so at least one robot stays active. Even if the user passes a
        # higher value we clamp here defensively.
        self.max_concurrent_grabs = max(0, min(max_concurrent_grabs, self.n_agents - 1))
        if init_regime_dist is None:
            init_regime_dist = DEFAULT_INIT_REGIME_DIST
        if len(init_regime_dist) != self.n_agents:
            raise ValueError(
                f"init_regime_dist must have length n_agents={self.n_agents}; "
                f"got {len(init_regime_dist)}"
            )
        s = float(sum(init_regime_dist))
        if s <= 0:
            raise ValueError("init_regime_dist must sum to a positive value")
        self.init_regime_dist = [p / s for p in init_regime_dist]
        self.init_hold_min = init_hold_min
        self.init_hold_max = init_hold_max
        self.rng = np.random.default_rng(seed)
        # per-env dict of {robot_idx -> _Grab}; a robot is teleop'd iff it's a key
        self.grabs: List[dict] = [{} for _ in range(self.num_envs)]
        # per-env step counter (kept for compat / debugging)
        self.steps = np.zeros(self.num_envs, dtype=np.int64)

    # ---- helpers --------------------------------------------------------
    def _sample_initial_active_count(self) -> int:
        """Sample active_count in {1, 2, ..., n_agents} from init_regime_dist."""
        return int(self.rng.choice(self.n_agents, p=self.init_regime_dist)) + 1

    def _start_grab(self, env_idx: int, robot: int, hold_still: bool = False):
        if hold_still:
            duration = int(self.rng.integers(self.init_hold_min, self.init_hold_max + 1))
            grab = _Grab(
                robot=robot, age=0, duration=duration,
                drift_dir=0.0, base_vy=0.0, drift_speed=0.0,
            )
        else:
            duration = int(self.rng.integers(self.min_duration, self.max_duration))
            drift_dir = 1.0 if self.rng.random() < 0.5 else -1.0
            base_vy = float(self.rng.uniform(0.0, 0.5))
            grab = _Grab(
                robot=robot, age=0, duration=duration,
                drift_dir=drift_dir, base_vy=base_vy, drift_speed=self.drift_speed,
            )
        self.grabs[env_idx][robot] = grab
        self.env.set_teleop(env_idx, robot, True)
        if hold_still:
            self.env.set_teleop_action(env_idx, robot, np.zeros(2, dtype=np.float32))

    def _release_grab(self, env_idx: int, robot: int):
        self.env.set_teleop(env_idx, robot, False)
        self.grabs[env_idx].pop(robot, None)

    # ---- public API -----------------------------------------------------
    def reset_env(self, env_idx: int):
        # release any holdovers from the previous episode
        for r in list(self.grabs[env_idx].keys()):
            self.env.set_teleop(env_idx, r, False)
        self.grabs[env_idx] = {}
        self.steps[env_idx] = 0

        # sample initial regime and pre-grab the robots that should be inactive
        active_count = self._sample_initial_active_count()
        n_grab = self.n_agents - active_count
        # also obey the concurrent-grab cap
        n_grab = min(n_grab, self.max_concurrent_grabs)
        if n_grab <= 0:
            return
        robots = self.rng.choice(self.n_agents, size=n_grab, replace=False)
        for r in robots:
            self._start_grab(env_idx, int(r), hold_still=True)

    def step(self):
        """Call once per env timestep, BEFORE env.vector_step.

        Updates teleop_mask + teleop_vels on the env in place.
        """
        for e in range(self.num_envs):
            self.steps[e] += 1

            # try to start new grabs on currently-untouched robots
            current = set(self.grabs[e].keys())
            slots_free = self.max_concurrent_grabs - len(current)
            if slots_free > 0:
                # iterate in a random order so we don't bias which robot gets grabbed
                order = list(range(self.n_agents))
                self.rng.shuffle(order)
                for r in order:
                    if slots_free <= 0:
                        break
                    if r in current:
                        continue
                    if self.rng.random() < self.p_grab:
                        self._start_grab(e, r, hold_still=False)
                        current.add(r)
                        slots_free -= 1

            # advance / release existing grabs and push their teleop velocities
            for r in list(self.grabs[e].keys()):
                grab = self.grabs[e][r]
                grab.age += 1
                if grab.age >= grab.duration or self.rng.random() < self.p_release:
                    self._release_grab(e, r)
                    continue
                phase = grab.age * 0.15
                vx = grab.drift_speed * grab.drift_dir * float(np.cos(phase))
                vy = grab.base_vy
                self.env.set_teleop_action(e, r, np.array([vx, vy], dtype=np.float32))


class KeyboardTeleop:
    """Pygame-key-driven teleop for the demo/eval window.

    Keys:
      1 / 2 / 3 / 4  toggle teleop on robot (1-indexed)
      W / A / S / D  drive the most-recently-selected teleop robot
      Z / X          decrease / increase teleop drive speed
      0              release all teleop robots
    """

    SPEED_STEP = 0.25
    MIN_DRIVE_SPEED = 0.25
    MAX_DRIVE_SPEED = 2.5

    DRIVE_KEYS = {
        ord("w"): np.array([0.0, 1.0]),
        ord("s"): np.array([0.0, -1.0]),
        ord("a"): np.array([-1.0, 0.0]),
        ord("d"): np.array([1.0, 0.0]),
    }

    def __init__(
        self,
        env,
        env_idx: int = 0,
        drive_speed: float = MAX_V,
        min_drive_speed: float = MIN_DRIVE_SPEED,
        max_drive_speed: float = MAX_DRIVE_SPEED,
        speed_step: float = SPEED_STEP,
    ):
        self.env = env
        self.env_idx = env_idx
        self.n_agents = env.cfg["n_agents"]
        self.min_drive_speed = min_drive_speed
        self.max_drive_speed = max_drive_speed
        self.speed_step = speed_step
        self.drive_speed = float(np.clip(drive_speed, min_drive_speed, max_drive_speed))
        self.selected = None  # last-toggled-on robot
        self.pressed = set()

    def _adjust_speed(self, delta: float):
        self.drive_speed = float(
            np.clip(
                self.drive_speed + delta,
                self.min_drive_speed,
                self.max_drive_speed,
            )
        )

    def _toggle(self, robot: int):
        active_now = float(self.env.teleop_mask[self.env_idx, robot]) > 0.5
        self.env.set_teleop(self.env_idx, robot, not active_now)
        if not active_now:
            self.selected = robot
        elif self.selected == robot:
            self.selected = None

    def handle_event(self, event):
        import pygame

        if event.type == pygame.KEYDOWN:
            if event.unicode in ("1", "2", "3", "4"):
                idx = int(event.unicode) - 1
                if 0 <= idx < self.n_agents:
                    self._toggle(idx)
            elif event.unicode == "0":
                for r in range(self.n_agents):
                    self.env.set_teleop(self.env_idx, r, False)
                self.selected = None
            elif event.key == pygame.K_x:
                self._adjust_speed(self.speed_step)
            elif event.key == pygame.K_z:
                self._adjust_speed(-self.speed_step)
            elif event.key in self.DRIVE_KEYS:
                self.pressed.add(event.key)
        elif event.type == pygame.KEYUP and event.key in self.DRIVE_KEYS:
            self.pressed.discard(event.key)

    def apply(self):
        """Compute desired velocity for the selected teleop robot from the
        current set of pressed keys and push it via set_teleop_action.

        Call once per frame, after pumping pygame events.
        """
        if self.selected is None:
            return
        v = np.zeros(2, dtype=np.float32)
        for k in self.pressed:
            v += self.DRIVE_KEYS[k]
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm * self.drive_speed
        # clear teleop velocities for any robot that was deselected since last frame
        for r in range(self.n_agents):
            if float(self.env.teleop_mask[self.env_idx, r]) > 0.5:
                if r == self.selected:
                    self.env.set_teleop_action(self.env_idx, r, v)
                else:
                    # held in place
                    self.env.set_teleop_action(self.env_idx, r, np.zeros(2, dtype=np.float32))


def _demo():
    """Run RandomTeleop against FakeHallwayEnv for a few steps and print masks.

    Lets you sanity-check the grab/release cadence without firing up the full env.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    from fake_env import FakeHallwayEnv

    env = FakeHallwayEnv(num_envs=2)
    env.vector_reset()
    rt = RandomTeleop(env, p_grab=0.05, p_release=0.05, seed=1)
    for e in range(env.cfg["num_envs"]):
        rt.reset_env(e)
    grabs_seen = 0
    for t in range(400):
        rt.step()
        env.vector_step(np.zeros((env.cfg["num_envs"], env.cfg["n_agents"], 2)))
        if rt.grabs[0]:
            grabs_seen += 1
    print(f"demo OK: 400 steps, env0 grabs active for {grabs_seen} steps")
    print(f"final teleop_mask=\n{env.teleop_mask}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    _demo()
