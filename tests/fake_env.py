"""Minimal stub satisfying the FormationHallwayEnv contract for parallel dev.

Lets Person 2 (teleop) and Person 3 (trainer) develop without the real env.
Returns shape-correct observations and accepts the teleop interface.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from contract import MAX_AGENTS, MAX_V


class FakeHallwayEnv:
    def __init__(self, num_envs: int = 1, n_agents: int = MAX_AGENTS):
        self.cfg = {
            "num_envs": num_envs,
            "n_agents": n_agents,
            "max_v": MAX_V,
            "dt": 0.05,
            "max_time_steps": 100,
        }
        self.teleop_mask = np.zeros((num_envs, n_agents), dtype=np.float32)
        self.present_mask = np.ones((num_envs, n_agents), dtype=np.float32)
        self.teleop_vels = np.zeros((num_envs, n_agents, 2), dtype=np.float32)
        self.ps = np.zeros((num_envs, n_agents, 2), dtype=np.float32)
        self.t = 0

    def _obs_one(self, e: int):
        n = self.cfg["n_agents"]
        return {
            "pos": [[0.0, 0.0]] * n,
            "vel": [[0.0, 0.0]] * n,
            "goal": [[0.0, 5.0]] * n,
            "teleop_mask": self.teleop_mask[e].tolist(),
            "present_mask": self.present_mask[e].tolist(),
            "time": [[self.t * 0.05]] * n,
        }

    def vector_reset(self):
        self.t = 0
        self.teleop_mask[:] = 0.0
        self.teleop_vels[:] = 0.0
        return [self._obs_one(e) for e in range(self.cfg["num_envs"])]

    def reset_at(self, e: int):
        self.teleop_mask[e] = 0.0
        self.teleop_vels[e] = 0.0
        return self._obs_one(e)

    def vector_step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        m = self.teleop_mask[..., None]
        actions = actions * (1 - m) + self.teleop_vels * m
        self.ps += actions * self.cfg["dt"]
        self.t += 1
        n = self.cfg["n_agents"]
        rewards_per_agent = np.random.normal(0.0, 0.1, size=(self.cfg["num_envs"], n))
        rewards_per_agent *= 1.0 - self.teleop_mask
        infos = [
            {"rewards": {k: float(rewards_per_agent[e, k]) for k in range(n)}}
            for e in range(self.cfg["num_envs"])
        ]
        dones = [self.t >= self.cfg["max_time_steps"]] * self.cfg["num_envs"]
        obs = [self._obs_one(e) for e in range(self.cfg["num_envs"])]
        return obs, rewards_per_agent.sum(axis=1).tolist(), dones, infos

    def set_teleop(self, env_idx: int, robot_idx: int, active: bool):
        self.teleop_mask[env_idx, robot_idx] = 1.0 if active else 0.0
        if not active:
            self.teleop_vels[env_idx, robot_idx] = 0.0

    def set_teleop_action(self, env_idx: int, robot_idx: int, vel):
        self.teleop_vels[env_idx, robot_idx] = np.clip(
            np.asarray(vel, dtype=np.float32), -self.cfg["max_v"], self.cfg["max_v"]
        )
