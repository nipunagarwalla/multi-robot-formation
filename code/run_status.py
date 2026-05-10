"""Single-run inspector for the auto-research loop.

Reads runs/<ts>/{iterations.csv, weights/best_eval.json} and prints a
structured key:value summary plus a verdict (RUNNING / DEPLOYABLE /
COLLAPSED / STALLED / DIVERGED / DONE / FAILED) and a one-line
recommendation. Every wake-up of the /loop driver runs this against the
current run dir and acts on the recommendation.

Output is line-oriented `key=value` so a future shell-script wrapper
could grep it; the verdict is on its own line for easy `tail -1`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional


# ---- thresholds (CLI-tunable) ----------------------------------------------

@dataclass
class Thresholds:
    warmup_iters: int = 200
    deploy_min_succ: float = 0.6
    deploy_max_form_err: float = 0.30
    diverge_max_score: float = -200.0
    diverge_min_iter: int = 300
    collapsed_max_entropy: float = -0.005    # > this means deterministic policy
    collapsed_consecutive_evals: int = 3
    collapsed_no_new_best_iters: int = 150
    stalled_min_score_gain: float = 50.0
    stalled_window_iters: int = 250
    expected_iterations: int = 1500


# ---- IO ---------------------------------------------------------------------

def _last_csv_rows(path: str, n: int) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[-n:]


def _read_best_eval(weights_dir: str) -> Optional[dict]:
    path = os.path.join(weights_dir, "best_eval.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _trainer_alive(run_dir: str, pid: Optional[int], stale_seconds: float = 90.0) -> bool:
    """True if the trainer subprocess is reachable.

    With --pid set, uses kill(pid, 0). Without one, falls back to "iterations.csv
    mtime within stale_seconds" — good enough for the inspector mode and
    (deliberately) treats long-finished runs as not-alive.
    """
    if pid is not None:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False
    iters_csv = os.path.join(run_dir, "iterations.csv")
    if not os.path.exists(iters_csv):
        return False
    age = abs(os.path.getmtime(iters_csv) - _now())
    return age < stale_seconds


def _now() -> float:
    import time
    return time.time()


# ---- scoring ---------------------------------------------------------------

def post_deploy_combined_score(per_n: dict) -> float:
    """Equal-weight goal-reaching + tight-circle score, range [0, 1]."""
    rs = list(per_n.values())
    if not rs:
        return 0.0
    succs = [r["success_rate"] for r in rs]
    forms = [r["mean_form_err"] for r in rs]
    goal_score = sum(succs) / len(succs)
    form_score = 1.0 - max(0.0, min(max(forms) / 0.5, 1.0))
    return (goal_score + form_score) / 2.0


# ---- verdict ---------------------------------------------------------------

VERDICTS = (
    "RUNNING", "DEPLOYABLE", "COLLAPSED", "STALLED", "DIVERGED", "DONE", "FAILED",
)


def classify(*, last_iter: int, latest_csv: list[dict], best_eval: Optional[dict],
             trainer_alive: bool, t: Thresholds) -> tuple[str, str]:
    """Return (verdict, reason)."""
    # Trainer dead?
    if not trainer_alive:
        if last_iter >= t.expected_iterations:
            return "DONE", f"reached --iterations {t.expected_iterations}"
        return "FAILED", f"trainer exited before reaching iter {t.expected_iterations} (last iter={last_iter})"

    # Pre-warmup: nothing to judge yet
    if last_iter < t.warmup_iters:
        return "RUNNING", f"warmup (iter {last_iter} < {t.warmup_iters})"

    # Deployable check
    if best_eval is not None:
        per_n = best_eval.get("per_n", {})
        if per_n:
            succs = [r["success_rate"] for r in per_n.values()]
            forms = [r["mean_form_err"] for r in per_n.values()]
            mean_form = sum(forms) / len(forms)
            if min(succs) >= t.deploy_min_succ and mean_form <= t.deploy_max_form_err:
                return "DEPLOYABLE", (
                    f"min_succ={min(succs):.2f} >= {t.deploy_min_succ}, "
                    f"mean_form={mean_form:.3f} <= {t.deploy_max_form_err}"
                )

    # Diverged
    score = (best_eval or {}).get("score")
    if score is not None and last_iter >= t.diverge_min_iter and score < t.diverge_max_score:
        return "DIVERGED", f"best score {score:.1f} < {t.diverge_max_score} after iter {last_iter}"

    # Collapsed: entropy near zero AND no new best in a while
    recent = latest_csv[-t.collapsed_consecutive_evals:]
    if (
        len(recent) >= t.collapsed_consecutive_evals
        and all(_safe_float(r.get("entropy")) > t.collapsed_max_entropy for r in recent)
    ):
        # Determine "no new best in N iters". The best.pt symlink target encodes
        # the iteration of the current best; compare with last_iter.
        best_iter = _best_iter_from_symlink(os.path.join(_inferred_weights_dir(latest_csv), "best.pt"))
        if best_iter is not None and (last_iter - best_iter) >= t.collapsed_no_new_best_iters:
            return "COLLAPSED", (
                f"entropy collapsed (last {t.collapsed_consecutive_evals} evals all > "
                f"{t.collapsed_max_entropy}) and no new best in {last_iter - best_iter} iters"
            )

    # Stalled: no significant best-score improvement in the window
    if score is not None:
        # Heuristic: read the score history from earlier eval json by walking
        # back through best.pt's timestamp would be ideal, but we don't have
        # that history. Approximate via "best_iter happened > stalled_window
        # iters ago AND best_score is below threshold". This is conservative.
        weights_dir = _inferred_weights_dir(latest_csv)
        best_iter = _best_iter_from_symlink(os.path.join(weights_dir, "best.pt"))
        if best_iter is not None and (last_iter - best_iter) >= t.stalled_window_iters:
            return "STALLED", (
                f"best.pt last updated at iter {best_iter}, no improvement in "
                f"{last_iter - best_iter} iters"
            )

    return "RUNNING", "no flag tripped"


def _safe_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _inferred_weights_dir(latest_csv: list[dict]) -> str:
    # latest_csv has no path info; the caller passes weights_dir separately
    # (this is a placeholder used by classify when rows are inspected without
    # a corresponding dir). Replaced by injection in main().
    return ""


def _best_iter_from_symlink(symlink_path: str) -> Optional[int]:
    """Resolve weights/best.pt -> weights_epoch{N}.pt and return N."""
    if not os.path.islink(symlink_path) and not os.path.exists(symlink_path):
        return None
    try:
        target = os.readlink(symlink_path) if os.path.islink(symlink_path) else os.path.basename(symlink_path)
    except OSError:
        return None
    name = os.path.basename(target)
    prefix, suffix = "weights_epoch", ".pt"
    if name.startswith(prefix) and name.endswith(suffix):
        try:
            return int(name[len(prefix):-len(suffix)])
        except ValueError:
            return None
    return None


# ---- main ------------------------------------------------------------------

def inspect(run_dir: str, *, pid: Optional[int], t: Thresholds) -> dict:
    """Read disk artifacts, classify, return a dict of summary fields."""
    iters_csv = os.path.join(run_dir, "iterations.csv")
    weights_dir = os.path.join(run_dir, "weights")
    rows = _last_csv_rows(iters_csv, n=max(10, t.collapsed_consecutive_evals))
    last_iter = int(rows[-1]["iter"]) if rows else 0
    last_row = rows[-1] if rows else {}
    best_eval = _read_best_eval(weights_dir)
    alive = _trainer_alive(run_dir, pid)

    # patch the helper to find weights/best.pt for kill-rule lookups
    global _inferred_weights_dir
    _orig = _inferred_weights_dir
    _inferred_weights_dir = lambda _rows, _wd=weights_dir: _wd
    try:
        verdict, reason = classify(
            last_iter=last_iter,
            latest_csv=rows,
            best_eval=best_eval,
            trainer_alive=alive,
            t=t,
        )
    finally:
        _inferred_weights_dir = _orig

    best_score = (best_eval or {}).get("score")
    per_n = (best_eval or {}).get("per_n", {})
    combined = post_deploy_combined_score(per_n) if per_n else 0.0
    best_iter = _best_iter_from_symlink(os.path.join(weights_dir, "best.pt"))

    summary = {
        "run_dir": run_dir,
        "pid": pid,
        "trainer_alive": alive,
        "last_iter": last_iter,
        "expected_iterations": t.expected_iterations,
        "wall_time_s": _safe_float(last_row.get("wall_time_s")),
        "mean_reward": _safe_float(last_row.get("mean_reward")),
        "entropy": _safe_float(last_row.get("entropy")),
        "approx_kl": _safe_float(last_row.get("approx_kl")),
        "clip_frac": _safe_float(last_row.get("clip_frac")),
        "iter_success_rate": _safe_float(last_row.get("success_rate")),
        "iter_mean_form": _safe_float(last_row.get("mean_formation_error")),
        "iter_mean_n_present": _safe_float(last_row.get("mean_n_present")),
        "best_score": best_score,
        "best_combined": combined,
        "best_iter": best_iter,
        "per_n": per_n,
        "verdict": verdict,
        "reason": reason,
        "recommendation": _recommend(verdict),
    }
    return summary


def _recommend(verdict: str) -> str:
    return {
        "RUNNING":    "continue",
        "DEPLOYABLE": "continue (snapshot best.pt; mark deployable; do not kill)",
        "COLLAPSED":  "kill: exploration collapsed",
        "STALLED":    "kill: best score not improving",
        "DIVERGED":   "kill: best score diverged",
        "DONE":       "advance: training completed normally",
        "FAILED":     "advance: trainer crashed",
    }.get(verdict, "manual review")


def print_summary(s: dict):
    """Line-oriented key=value output, verdict on its own line at the end."""
    keys = (
        "run_dir", "pid", "trainer_alive", "last_iter", "expected_iterations",
        "wall_time_s", "mean_reward", "entropy", "approx_kl", "clip_frac",
        "iter_success_rate", "iter_mean_form", "iter_mean_n_present",
        "best_score", "best_combined", "best_iter",
    )
    for k in keys:
        v = s[k]
        if isinstance(v, float):
            if math.isnan(v):
                v_str = "nan"
            else:
                v_str = f"{v:.4f}"
        else:
            v_str = str(v)
        print(f"{k}={v_str}")
    if s["per_n"]:
        for n, r in s["per_n"].items():
            print(
                f"per_n.{n}=succ={r['success_rate']*100:.0f}% "
                f"v_y={r['mean_v_y']:+.3f} form={r['mean_form_err']:.3f} "
                f"r_circ={r['mean_circle_radius']:.2f}"
            )
    print(f"reason={s['reason']}")
    print(f"recommendation={s['recommendation']}")
    print(f"verdict={s['verdict']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="path to the runs/<ts>_<tag> directory")
    ap.add_argument("--pid", type=int, default=None,
                    help="trainer PID; if alive trainer_alive=True")
    ap.add_argument("--warmup-iters", type=int, default=200)
    ap.add_argument("--deploy-min-succ", type=float, default=0.6)
    ap.add_argument("--deploy-max-form-err", type=float, default=0.30)
    ap.add_argument("--expected-iterations", type=int, default=1500)
    ap.add_argument("--json", action="store_true",
                    help="emit one JSON object instead of key=value lines")
    args = ap.parse_args()

    t = Thresholds(
        warmup_iters=args.warmup_iters,
        deploy_min_succ=args.deploy_min_succ,
        deploy_max_form_err=args.deploy_max_form_err,
        expected_iterations=args.expected_iterations,
    )
    if not os.path.isdir(args.run_dir):
        print(f"verdict=FAILED\nreason=run dir does not exist: {args.run_dir}", file=sys.stderr)
        sys.exit(2)
    summary = inspect(args.run_dir, pid=args.pid, t=t)
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print_summary(summary)


if __name__ == "__main__":
    main()
