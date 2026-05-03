"""Print + plot side-by-side training-run comparison.

Reads N run directories under runs/ and emits:
  - a sortable summary table to stdout
  - runs/_comparison/<timestamp>.png with mean_reward and policy_loss
    over iterations, one curve per run

Plots are best-effort — if matplotlib isn't installed the printed table
still works, which is the must-have artefact.

Usage:
  python code/scripts/compare_runs.py runs/<ts>_v0 runs/<ts>_v1
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
from typing import Any, Dict, List


def read_iterations(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, "iterations.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in list(row.items()):
                if v == "":
                    row[k] = None
                else:
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            rows.append(row)
    return rows


def read_episodes(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, "episodes.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def summarise(run_dir: str) -> Dict[str, Any]:
    iters = read_iterations(run_dir)
    eps = read_episodes(run_dir)
    out: Dict[str, Any] = {"run": os.path.basename(run_dir), "iterations": len(iters)}
    if iters:
        out["last_mean_reward"] = iters[-1].get("mean_reward")
        out["last_policy_loss"] = iters[-1].get("policy_loss")
        out["last_value_loss"] = iters[-1].get("value_loss")
        out["last_kl"] = iters[-1].get("approx_kl")
        out["wall_time_s"] = iters[-1].get("wall_time_s")
    out["episodes"] = len(eps)
    if eps:
        out["mean_episode_reward"] = sum(e["total_reward"] for e in eps) / len(eps)
        out["success_rate"] = sum(1 for e in eps if e.get("reached_goal")) / len(eps)
        out["mean_collisions"] = sum(e["num_collisions"] for e in eps) / len(eps)
    return out


def print_table(summaries: List[Dict[str, Any]]):
    cols = [
        "run",
        "iterations",
        "episodes",
        "last_mean_reward",
        "last_policy_loss",
        "last_value_loss",
        "last_kl",
        "mean_episode_reward",
        "success_rate",
        "mean_collisions",
        "wall_time_s",
    ]
    widths = {c: max(len(c), max((len(_fmt(s.get(c))) for s in summaries), default=0)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for s in summaries:
        print("  ".join(_fmt(s.get(c)).ljust(widths[c]) for c in cols))


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def plot(summaries_with_iters, out_path: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[compare] matplotlib not installed; skipping plot at {out_path}")
        return False

    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(11, 4))
    for run_name, iters in summaries_with_iters:
        if not iters:
            continue
        xs = [r["iter"] for r in iters]
        ax_r.plot(xs, [r.get("mean_reward") or 0 for r in iters], label=run_name)
        ax_l.plot(xs, [r.get("policy_loss") or 0 for r in iters], label=run_name)
    ax_r.set_title("mean_reward / iter")
    ax_r.set_xlabel("iter")
    ax_l.set_title("policy_loss / iter")
    ax_l.set_xlabel("iter")
    ax_r.legend(fontsize=8)
    ax_l.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories under runs/")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    summaries = [summarise(r) for r in args.runs]
    iters_by_run = [(os.path.basename(r), read_iterations(r)) for r in args.runs]
    print_table(summaries)

    if args.no_plot:
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = os.path.dirname(os.path.abspath(args.runs[0]))
    out_path = os.path.join(parent, "_comparison", f"{ts}.png")
    if plot(iters_by_run, out_path):
        print(f"[compare] plot -> {out_path}")


if __name__ == "__main__":
    main()
