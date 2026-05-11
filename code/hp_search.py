"""Hyperparam search space for the auto-research loop.

Two halves:
  * iter_grid()   yields the 8 hand-priority configs first (highest prior
                  from the v3 history), then defers to random sampling.
  * iter_random() yields uniform samples from the broader grid.

next_config(state_path) is the orchestrator's pull-next-thing-to-try API:
reads tried_config_ids from state.json, returns the next un-tried config,
and updates state. Self-contained — no imports from the trainer.

Frozen knobs (env geometry, agent radius, kinematics, reward coeffs)
NEVER appear here. The presence of any frozen knob in SEARCH_SPACE is
caught by test_no_frozen_knobs in tests/.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
from typing import Iterator


# Knob -> list of values. Random-fallback samples from this space.
# PPO knobs are anchored at cfg #1's values for the v2 sweep (the prior
# loop showed cfg #1 dominated; iterating reward shaping is the new lever),
# but we keep them in the space so random sampling can still wander.
SEARCH_SPACE = {
    # PPO (anchored in HAND_PRIORITY but tunable in random fallback)
    "lr": [5e-5, 1e-4, 2e-4],
    "entropy_coeff": [0.03, 0.05, 0.08],
    "vf_loss_coeff": [0.25, 0.5, 1.0],
    "max_grad_norm": [0.5, 1.0],
    "num_sgd_iter": [4, 8],
    "gamma": [0.99, 0.995],
    "p_grab": [0.003, 0.005],
    "init_n_present_dist": ["flat", "easy", "hard"],
    # Reward shaping (the v2 lever)
    "k_fwd":   [5.0, 10.0, 20.0],
    "k_form":  [0.5, 2.0, 5.0],
    "k_coll":  [2.0, 5.0],
    "k_wall":  [0.5, 1.0],
    "k_goal":  [20.0, 50.0, 100.0],
    "k_stall": [0.0, 0.5, 2.0],
}


# Named init_n_present_dist presets serialised as 10-element comma-joined floats.
INIT_N_PRESENT_DIST_PRESETS = {
    "flat": [1.0] * 10,
    "easy": [0.5, 0.7, 1.0, 1.5, 1.5, 1.0, 0.7, 0.5, 0.3, 0.3],
    "hard": [0.3, 0.3, 0.5, 0.7, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5],
}


# v2 sweep: anchor PPO knobs at cfg #1's values (the prior loop's
# best-by-best-score config) and vary reward shaping. v1 showed every
# entropy/lr combination collapsed; reward coefs are the remaining lever.
ANCHOR_PPO = {
    "lr": 1e-4, "entropy_coeff": 0.05, "vf_loss_coeff": 0.5,
    "max_grad_norm": 1.0, "num_sgd_iter": 8, "gamma": 0.995,
    "p_grab": 0.005, "init_n_present_dist": "flat",
}

# Hypothesis-driven reward-coef variants:
#   1. k_goal heavy (5x) — make goal-reach dominate; everything else baseline
#   2. k_form light (1/4) — let the policy learn to walk before circling
#   3. k_fwd heavy (2x) — bigger forward-y gradient signal
#   4. Combined 1+2+3
#   5. No stall — don't punish standing still (was masking real progress signal)
#   6. Aggressive — max forward, max goal, min penalties (last-ditch shaping)
HAND_PRIORITY = [
    {**ANCHOR_PPO, "k_fwd": 5.0,  "k_form": 2.0, "k_coll": 5.0, "k_wall": 1.0, "k_goal": 100.0, "k_stall": 0.5},
    {**ANCHOR_PPO, "k_fwd": 5.0,  "k_form": 0.5, "k_coll": 5.0, "k_wall": 1.0, "k_goal": 20.0,  "k_stall": 0.5},
    {**ANCHOR_PPO, "k_fwd": 10.0, "k_form": 2.0, "k_coll": 5.0, "k_wall": 1.0, "k_goal": 20.0,  "k_stall": 0.5},
    {**ANCHOR_PPO, "k_fwd": 10.0, "k_form": 0.5, "k_coll": 5.0, "k_wall": 1.0, "k_goal": 100.0, "k_stall": 0.5},
    {**ANCHOR_PPO, "k_fwd": 5.0,  "k_form": 2.0, "k_coll": 5.0, "k_wall": 1.0, "k_goal": 20.0,  "k_stall": 0.0},
    {**ANCHOR_PPO, "k_fwd": 20.0, "k_form": 0.5, "k_coll": 2.0, "k_wall": 0.5, "k_goal": 100.0, "k_stall": 0.0},
]


def _config_id(cfg: dict) -> str:
    """Deterministic short id from the sorted config items."""
    s = json.dumps(cfg, sort_keys=True)
    return "cfg_" + hashlib.sha1(s.encode()).hexdigest()[:8]


def iter_random(seed: int = 0) -> Iterator[dict]:
    """Yield random samples from SEARCH_SPACE forever (caller breaks)."""
    rng = random.Random(seed)
    keys = list(SEARCH_SPACE.keys())
    while True:
        yield {k: rng.choice(SEARCH_SPACE[k]) for k in keys}


def iter_grid(random_seed: int = 0, max_random: int = 30) -> Iterator[dict]:
    """Yield hand-priority configs, then up to max_random random samples."""
    yield from HAND_PRIORITY
    seen_ids = {_config_id(c) for c in HAND_PRIORITY}
    rng_iter = iter_random(seed=random_seed)
    emitted = 0
    while emitted < max_random:
        cfg = next(rng_iter)
        cid = _config_id(cfg)
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        emitted += 1
        yield cfg


def as_cli_args(cfg: dict) -> list[str]:
    """Serialize one config as trainer-CLI flags."""
    args: list[str] = []
    for k, v in cfg.items():
        if k == "init_n_present_dist":
            preset = INIT_N_PRESENT_DIST_PRESETS[v]
            args += ["--init-n-present-dist", ",".join(str(x) for x in preset)]
        else:
            args += [f"--{k.replace('_', '-')}", str(v)]
    return args


def next_config(state_path: str, max_random: int = 30) -> tuple[str, dict] | None:
    """Pop the next un-tried config given the current state.json.

    Returns (config_id, config) or None if the search space is exhausted.
    Caller is responsible for appending to state.tried_config_ids before
    starting the next run.
    """
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    else:
        state = {}
    tried = set(state.get("tried_config_ids", []))
    for cfg in iter_grid(random_seed=state.get("random_seed", 0), max_random=max_random):
        cid = _config_id(cfg)
        if cid not in tried:
            return cid, cfg
    return None


def _dry_run(n_random_to_show: int = 5):
    print(f"# Hand-priority configs ({len(HAND_PRIORITY)}):")
    for cfg in HAND_PRIORITY:
        print(f"  {_config_id(cfg)}: {json.dumps(cfg, sort_keys=True)}")
    print(f"\n# First {n_random_to_show} random samples (seed=0):")
    seen = {_config_id(c) for c in HAND_PRIORITY}
    rng_iter = iter_random(seed=0)
    shown = 0
    while shown < n_random_to_show:
        cfg = next(rng_iter)
        cid = _config_id(cfg)
        if cid in seen:
            continue
        seen.add(cid)
        shown += 1
        print(f"  {cid}: {json.dumps(cfg, sort_keys=True)}")
    print(f"\n# Example as_cli_args() for first hand-priority config:")
    print("  " + " ".join(as_cli_args(HAND_PRIORITY[0])))


def main():
    ap = argparse.ArgumentParser(
        description="Hyperparam search space for the auto-research loop."
    )
    sub = ap.add_subparsers(dest="cmd")

    sp_dry = sub.add_parser("dry-run", help="print upcoming configs")
    sp_dry.add_argument("--n-random", type=int, default=5)

    sp_next = sub.add_parser("next", help="pop the next config given a state.json")
    sp_next.add_argument("--state", required=True, help="path to state.json")
    sp_next.add_argument("--max-random", type=int, default=30)

    # also accept --dry-run as a shortcut (no subcommand)
    ap.add_argument("--dry-run", action="store_true",
                    help="shortcut for the dry-run subcommand")
    args = ap.parse_args()

    if args.cmd == "dry-run" or args.dry_run:
        _dry_run(getattr(args, "n_random", 5))
        return
    if args.cmd == "next":
        result = next_config(args.state, max_random=args.max_random)
        if result is None:
            print(json.dumps({"exhausted": True}))
        else:
            cid, cfg = result
            print(json.dumps({"config_id": cid, "config": cfg, "cli_args": as_cli_args(cfg)}))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
