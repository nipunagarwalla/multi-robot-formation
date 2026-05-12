#!/usr/bin/env python3
"""
Write a randomly-initialized circle_policy_v1 checkpoint to disk so the
launch + circle_node pipeline can be smoke-tested without a real trained
.pt file.

The checkpoint format matches code/checkpoint.py exactly:
    {"agent": state_dict, "optimizer": None, "iteration": 0}

What the policy will do at runtime with these random weights: produce
samples from Beta distributions whose alpha/beta come from a freshly
initialized GNN. Robots will jitter and drift around without doing
anything meaningful — but every code path is exercised, and the .pt
file loads cleanly through checkpoint.load_checkpoint().

Usage:
    python3 ros2/limo_circle_sim/scripts/make_fake_checkpoint.py
        [--out PATH]   default: <repo>/weights/fake.pt
        [--seed N]     default: 0
"""
from __future__ import annotations

import argparse
import pathlib
import sys


def main() -> None:
    here = pathlib.Path(__file__).resolve()
    repo = here.parents[3]  # ros2/limo_circle_sim/scripts/this -> repo root
    code_dir = repo / "code"
    if not code_dir.is_dir():
        raise RuntimeError(f"could not find policy code at {code_dir}")
    sys.path.insert(0, str(code_dir))

    import gymnasium as gym  # noqa: E402
    import torch  # noqa: E402

    from contract import DT, MAX_AGENTS, MAX_V, WORLD_H  # noqa: E402
    from model import Agent  # noqa: E402
    from checkpoint import save_checkpoint  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(repo / "weights" / "fake.pt"),
        help="path to write the .pt file (default: <repo>/weights/fake.pt)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # Same env shim circle_node.py uses — just enough surface for Agent.__init__.
    class Shim:
        pass

    shim = Shim()
    n = MAX_AGENTS
    shim.cfg = {"n_agents": n, "num_envs": 1, "dt": DT, "max_v": MAX_V}
    shim.action_space = gym.spaces.Tuple(
        (gym.spaces.Box(low=-MAX_V, high=MAX_V, shape=(2,), dtype=float),) * n
    )
    max_t = 600 * DT
    shim.observation_space = gym.spaces.Dict({
        "pos":          gym.spaces.Box(-WORLD_H, WORLD_H, shape=(n, 2), dtype=float),
        "vel":          gym.spaces.Box(-1e5, 1e5,         shape=(n, 2), dtype=float),
        "goal":         gym.spaces.Box(-WORLD_H, WORLD_H, shape=(n, 2), dtype=float),
        "teleop_mask":  gym.spaces.Box(0.0, 1.0,          shape=(n,),   dtype=float),
        "present_mask": gym.spaces.Box(0.0, 1.0,          shape=(n,),   dtype=float),
        "time":         gym.spaces.Box(0.0, max_t,        shape=(n, 1), dtype=float),
    })

    model_cfg = {
        "model": {
            "custom_model_config": {
                "activation": "relu",
                "msg_features": 32,
                "comm_range": 4.0,
                "use_masks": True,
            }
        }
    }
    agent = Agent(shim, model_cfg)
    n_params = sum(p.numel() for p in agent.parameters())

    out = pathlib.Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(out), agent, None, 0)
    print(f"wrote {out}  ({n_params} params, seed={args.seed})")
    print("Note: random-init policy — robots will jitter, not solve the task.")


if __name__ == "__main__":
    main()
