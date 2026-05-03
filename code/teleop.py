"""Teleop drivers for FormationHallwayEnv.

RandomTeleop  — synthetic disturbance for training.
KeyboardTeleop — pygame key handler for the eval/demo binary.

Both call env.set_teleop(env_idx, robot_idx, active) and
env.set_teleop_action(env_idx, robot_idx, vel) — see code/contract.py.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contract import MAX_AGENTS, MAX_V


@dataclass
class _Grab:
    robot: int
    age: int
    duration: int
    drift_dir: float  # +1 / -1 lateral push
    base_vy: float
    drift_speed: float


class RandomTeleop:
    """Per-env synthetic teleop: occasionally grab a random robot, push it on
    a sinusoidal lateral trajectory away from the cluster, then release.

    Mirrors what a human will do during inference so the policy learns to
    re-form around a missing teammate and re-absorb returning robots.
    """

    def __init__(
        self,
        env,
        p_grab: float = 0.005,
        p_release: float = 0.01,
        drift_speed: float = 0.6,
        min_duration: int = 40,
        max_duration: int = 160,
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
        self.rng = np.random.default_rng(seed)
        # one active grab per env (at most one teleop'd robot at a time during training)
        self.grabs = [None for _ in range(self.num_envs)]
        # per-env counter for the sinusoid phase
        self.steps = np.zeros(self.num_envs, dtype=np.int64)

    def reset_env(self, env_idx: int):
        if self.grabs[env_idx] is not None:
            self.env.set_teleop(env_idx, self.grabs[env_idx].robot, False)
        self.grabs[env_idx] = None
        self.steps[env_idx] = 0

    def step(self):
        """Call once per env timestep, BEFORE env.vector_step.

        Updates teleop_mask + teleop_vels on the env in place.
        """
        for e in range(self.num_envs):
            self.steps[e] += 1
            grab = self.grabs[e]
            if grab is None:
                if self.rng.random() < self.p_grab:
                    robot = int(self.rng.integers(0, self.n_agents))
                    duration = int(self.rng.integers(self.min_duration, self.max_duration))
                    drift_dir = 1.0 if self.rng.random() < 0.5 else -1.0
                    base_vy = float(self.rng.uniform(0.0, 0.5))
                    self.grabs[e] = _Grab(
                        robot=robot,
                        age=0,
                        duration=duration,
                        drift_dir=drift_dir,
                        base_vy=base_vy,
                        drift_speed=self.drift_speed,
                    )
                    self.env.set_teleop(e, robot, True)
                    grab = self.grabs[e]
            if grab is None:
                continue
            grab.age += 1
            if grab.age >= grab.duration or self.rng.random() < self.p_release:
                self.env.set_teleop(e, grab.robot, False)
                self.grabs[e] = None
                continue
            phase = grab.age * 0.15
            vx = grab.drift_speed * grab.drift_dir * float(np.cos(phase))
            vy = grab.base_vy
            self.env.set_teleop_action(e, grab.robot, np.array([vx, vy], dtype=np.float32))


class KeyboardTeleop:
    """Pygame-key-driven teleop for the demo/eval window.

    Keys:
      1 / 2 / 3 / 4  toggle teleop on robot (1-indexed)
      W / A / S / D  drive the most-recently-selected teleop robot
      0              release all teleop robots
    """

    DRIVE_KEYS = {
        ord("w"): np.array([0.0, 1.0]),
        ord("s"): np.array([0.0, -1.0]),
        ord("a"): np.array([-1.0, 0.0]),
        ord("d"): np.array([1.0, 0.0]),
    }

    def __init__(self, env, env_idx: int = 0, drive_speed: float = MAX_V):
        self.env = env
        self.env_idx = env_idx
        self.n_agents = env.cfg["n_agents"]
        self.drive_speed = drive_speed
        self.selected = None  # last-toggled-on robot
        self.pressed = set()

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
    grabs_seen = 0
    for t in range(400):
        rt.step()
        env.vector_step(np.zeros((env.cfg["num_envs"], env.cfg["n_agents"], 2)))
        if rt.grabs[0] is not None:
            grabs_seen += 1
    print(f"demo OK: 400 steps, env0 grabs active for {grabs_seen} steps")
    print(f"final teleop_mask=\n{env.teleop_mask}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    _demo()
