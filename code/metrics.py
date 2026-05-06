"""Persistent training metrics for cross-run quantitative comparison.

Per Section 4.5 of plan.md every run gets its own directory under runs/
with three append-only files:

  config.json    — hyperparams, reward coeffs, env constants, git SHA
  iterations.csv — one row per PPO iteration
  episodes.jsonl — one JSON object per finished episode

Plus weights/weights_epoch{i}.pt + weights/latest.pt symlink.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import platform
import socket
import subprocess
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


ITER_COLUMNS: List[str] = [
    "iter",
    "wall_time_s",
    "env_steps",
    "policy_loss",
    "value_loss",
    "entropy",
    "total_loss",
    "approx_kl",
    "clip_frac",
    "grad_norm",
    "mean_reward",
    "mean_episode_length",
    "lr",
    # success / regime breakdown — populated per iteration from finished episodes
    "success_rate",
    "formation_error_active_1",
    "formation_error_active_2",
    "formation_error_active_3",
    "formation_error_active_4",
    "n_episodes_active_1",
    "n_episodes_active_2",
    "n_episodes_active_3",
    "n_episodes_active_4",
]


def _git_sha(repo_root: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _torch_versions() -> Dict[str, Any]:
    try:
        import torch

        return {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
        }
    except Exception:
        return {}


class RunLogger:
    def __init__(self, root_dir: str, tag: str = "run", repo_root: Optional[str] = None):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{ts}_{tag}"
        self.dir = os.path.join(root_dir, self.run_id)
        self.weights_dir = os.path.join(self.dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        self.iter_path = os.path.join(self.dir, "iterations.csv")
        self.ep_path = os.path.join(self.dir, "episodes.jsonl")
        self.eval_path = os.path.join(self.dir, "evals.jsonl")
        self.best_eval_path = os.path.join(self.dir, "best_eval.json")
        self.cfg_path = os.path.join(self.dir, "config.json")
        self._iter_writer = None
        self._iter_fh = None
        self._ep_fh = None
        self._eval_fh = None
        self._repo_root = repo_root or os.getcwd()

    # ---- one-shot config ----------------------------------------------
    def write_config(self, payload: Dict[str, Any]):
        meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "git_sha": _git_sha(self._repo_root),
            **_torch_versions(),
        }
        out = {"meta": meta, **payload}
        with open(self.cfg_path, "w") as f:
            json.dump(out, f, indent=2, default=str)

    # ---- per-iteration ------------------------------------------------
    def _ensure_iter(self):
        if self._iter_writer is None:
            self._iter_fh = open(self.iter_path, "a", newline="")
            self._iter_writer = csv.DictWriter(self._iter_fh, fieldnames=ITER_COLUMNS)
            if self._iter_fh.tell() == 0:
                self._iter_writer.writeheader()

    def log_iter(self, **fields):
        self._ensure_iter()
        row = {k: fields.get(k, "") for k in ITER_COLUMNS}
        self._iter_writer.writerow(row)
        self._iter_fh.flush()

    # ---- per-episode --------------------------------------------------
    def log_episode(self, payload: Dict[str, Any]):
        if self._ep_fh is None:
            self._ep_fh = open(self.ep_path, "a")
        self._ep_fh.write(json.dumps(payload, default=str) + "\n")
        self._ep_fh.flush()

    # ---- periodic eval -------------------------------------------------
    def log_eval(self, payload: Dict[str, Any]):
        if self._eval_fh is None:
            self._eval_fh = open(self.eval_path, "a")
        self._eval_fh.write(json.dumps(payload, default=str) + "\n")
        self._eval_fh.flush()

    def write_best_eval(self, payload: Dict[str, Any]):
        with open(self.best_eval_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    # ---- checkpoints --------------------------------------------------
    def checkpoint_path(self, iteration: int) -> str:
        return os.path.join(self.weights_dir, f"weights_epoch{iteration}.pt")

    def _update_weight_symlink(self, name: str, ckpt_path: str):
        link = os.path.join(self.weights_dir, name)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        # Use relative target so the symlink survives a moved runs/ tree
        rel = os.path.relpath(ckpt_path, self.weights_dir)
        os.symlink(rel, link)

    def update_latest_symlink(self, ckpt_path: str):
        self._update_weight_symlink("latest.pt", ckpt_path)

    def update_best_symlink(self, ckpt_path: str):
        self._update_weight_symlink("best.pt", ckpt_path)

    def close(self):
        if self._iter_fh is not None:
            self._iter_fh.close()
            self._iter_fh = None
            self._iter_writer = None
        if self._ep_fh is not None:
            self._ep_fh.close()
            self._ep_fh = None
        if self._eval_fh is not None:
            self._eval_fh.close()
            self._eval_fh = None


class EpisodeAccumulator:
    """Per-env counters that reset whenever the env resets.

    Trainer instantiates one per parallel env, calls update() each step,
    and emits() to RunLogger when done fires.
    """

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.length = 0
        self.total_reward = 0.0
        self.num_collisions = 0
        self.num_wall_hits = 0
        self.num_teleop_grabs = 0
        self.max_active = 0
        self.min_active = self.n_agents
        self.formation_err_sum = 0.0
        self.formation_err_count = 0
        self.formation_err_by_active: Dict[int, List[float]] = defaultdict(list)
        self.fwd_velocity_sum = 0.0
        self.fwd_velocity_count = 0
        self.stall_steps = 0
        self.last_teleop_mask: Optional[List[float]] = None

    def update(
        self,
        per_agent_rewards: Iterable[float],
        active_count: int,
        teleop_mask: Iterable[float],
        formation_err: Optional[float] = None,
        fwd_velocity: Optional[float] = None,
        stalled: bool = False,
        had_collision: bool = False,
        had_wall_hit: bool = False,
    ):
        self.length += 1
        rewards_l = list(per_agent_rewards)
        self.total_reward += sum(rewards_l)
        if had_collision:
            self.num_collisions += 1
        if had_wall_hit:
            self.num_wall_hits += 1
        self.max_active = max(self.max_active, active_count)
        self.min_active = min(self.min_active, active_count)
        if formation_err is not None and active_count >= 2:
            self.formation_err_sum += formation_err
            self.formation_err_count += 1
            self.formation_err_by_active[active_count].append(formation_err)
        if fwd_velocity is not None:
            self.fwd_velocity_sum += fwd_velocity
            self.fwd_velocity_count += 1
        if stalled:
            self.stall_steps += 1
        # detect grab edges (0 -> 1 transitions)
        m = list(teleop_mask)
        if self.last_teleop_mask is not None:
            for i, (prev, curr) in enumerate(zip(self.last_teleop_mask, m)):
                if prev < 0.5 and curr >= 0.5:
                    self.num_teleop_grabs += 1
        self.last_teleop_mask = m

    def emit(self, iteration: int, env_id: int, reached_goal: bool) -> Dict[str, Any]:
        per_active = {
            str(k): float(sum(v) / max(len(v), 1))
            for k, v in self.formation_err_by_active.items()
        }
        out = {
            "iter": iteration,
            "env_id": env_id,
            "episode_length": self.length,
            "total_reward": self.total_reward,
            "reached_goal": bool(reached_goal),
            "num_collisions": self.num_collisions,
            "num_wall_hits": self.num_wall_hits,
            "num_teleop_grabs": self.num_teleop_grabs,
            "max_active_count": self.max_active,
            "min_active_count": self.min_active,
            "formation_error_mean": (
                self.formation_err_sum / max(self.formation_err_count, 1)
            ),
            "formation_error_per_active_count": per_active,
            "forward_velocity_mean": (
                self.fwd_velocity_sum / max(self.fwd_velocity_count, 1)
            ),
            "stall_steps": self.stall_steps,
        }
        return out
