"""One wake-up of the auto-research loop. Run from cron every N min.

Self-contained: reads state.json, inspects current run, applies the
convergence rules, takes mechanical action (kill / advance / snapshot
best), writes state + journal. Prints a structured summary so Claude
(when driving via /loop) can read it and intervene on edge cases.

Designed to be the only command invoked per wake-up, so the whole
recurring loop sits behind a single `.venv/bin/python *` allow rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hp_search
import run_status


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(session_dir: str) -> dict:
    with open(os.path.join(session_dir, "state.json")) as f:
        return json.load(f)


def save_state(session_dir: str, state: dict):
    with open(os.path.join(session_dir, "state.json"), "w") as f:
        json.dump(state, f, indent=2)


def append_journal(session_dir: str, entry: dict):
    entry.setdefault("ts", time.time())
    with open(os.path.join(session_dir, "journal.jsonl"), "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def proc_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def kill_run(pid: int, grace_seconds: float = 10.0):
    """SIGTERM, wait up to grace_seconds, SIGKILL if still alive."""
    if not proc_alive(pid):
        return "already_dead"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_dead"
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if not proc_alive(pid):
            return "sigterm"
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return "sigkill"


def start_run(*, session_dir: str, config_id: str, config: dict, tag_prefix: str,
              iterations: int, num_envs: int, max_steps: int,
              checkpoint_every: int, eval_every: int, eval_episodes: int,
              eval_n_present_counts: str) -> tuple[int, str]:
    """Start a training subprocess in the background. Returns (pid, run_dir)."""
    tag = f"{tag_prefix}-{config_id}"
    log_path = os.path.join(session_dir, f"{config_id}.log")
    cmd = [
        os.path.join(REPO_ROOT, ".venv/bin/python"), "-u",
        os.path.join(HERE, "train_hallway.py"),
        "--iterations", str(iterations),
        "--num-envs", str(num_envs),
        "--max-steps", str(max_steps),
        "--checkpoint-every", str(checkpoint_every),
        "--eval-every", str(eval_every),
        "--eval-episodes", str(eval_episodes),
        "--eval-n-present-counts", eval_n_present_counts,
        "--tag", tag,
    ] + hp_search.as_cli_args(config)
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh, stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
        start_new_session=True,  # detach so it survives this script
    )
    # wait briefly for the trainer to print its run dir
    for _ in range(30):
        time.sleep(0.5)
        run_dir = _find_run_dir(tag)
        if run_dir is not None:
            return proc.pid, run_dir
    raise RuntimeError(f"trainer did not announce run dir within 15 s (tag={tag})")


def _find_run_dir(tag: str) -> Optional[str]:
    runs_root = os.path.join(REPO_ROOT, "runs")
    candidates = sorted(
        (d for d in os.listdir(runs_root) if d.endswith("_" + tag)),
        reverse=True,
    )
    if not candidates:
        return None
    return os.path.join("runs", candidates[0])


def update_current_best(session_dir: str, run_dir: str, summary: dict, state: dict):
    """Point session/current_best.pt at run_dir/weights/best.pt and copy eval json."""
    weights_dir = os.path.join(run_dir, "weights")
    best_pt = os.path.join(weights_dir, "best.pt")
    if not os.path.exists(best_pt):
        return False
    link = os.path.join(session_dir, "current_best.pt")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    rel = os.path.relpath(best_pt, session_dir)
    os.symlink(rel, link)
    src_json = os.path.join(weights_dir, "best_eval.json")
    if os.path.exists(src_json):
        shutil.copyfile(src_json, os.path.join(session_dir, "current_best_eval.json"))
    state["current_best"] = {
        "config_id": state["current_run"]["config_id"],
        "run_dir": run_dir,
        "score": summary["best_score"],
        "combined": summary["best_combined"],
        "iter": summary["best_iter"],
    }
    return True


def is_deployable(summary: dict, t: run_status.Thresholds) -> bool:
    per_n = summary.get("per_n") or {}
    if not per_n:
        return False
    succs = [r["success_rate"] for r in per_n.values()]
    forms = [r["mean_form_err"] for r in per_n.values()]
    if not succs:
        return False
    mean_form = sum(forms) / len(forms)
    return min(succs) >= t.deploy_min_succ and mean_form <= t.deploy_max_form_err


def end_session(session_dir: str, state: dict, *, reason: str, event: str = "STOPPED"):
    state["session_status"] = "stopped"
    state["stop_reason"] = reason
    state["stopped_ts"] = time.time()
    save_state(session_dir, state)
    append_journal(session_dir, {"event": event, "reason": reason})


# -- main wake-up flow -------------------------------------------------------

def tick(session_dir: str, *, train_iterations: int, train_num_envs: int,
         train_max_steps: int, train_checkpoint_every: int, train_eval_every: int,
         train_eval_episodes: int, train_eval_n_present_counts: str,
         tag_prefix: str, t: run_status.Thresholds):
    state = load_state(session_dir)
    if state.get("session_status") == "stopped":
        print("STATUS=stopped reason=" + state.get("stop_reason", "unknown"))
        return

    contract_path = os.path.join(REPO_ROOT, "code", "contract.py")
    sha_now = sha256(contract_path)
    if sha_now != state["contract_sha_at_start"]:
        end_session(
            session_dir, state,
            reason=f"contract.py SHA changed ({state['contract_sha_at_start'][:8]} -> {sha_now[:8]})",
            event="STOPPED",
        )
        print("STATUS=stopped reason=contract_changed")
        return

    now = time.time()
    elapsed_h = (now - state["session_start_ts"]) / 3600.0
    state["elapsed_hours"] = elapsed_h
    if elapsed_h > state["budget_hours"]:
        end_session(
            session_dir, state,
            reason=f"budget exceeded ({elapsed_h:.2f} > {state['budget_hours']} h)",
            event="FINAL_SUMMARY",
        )
        print(f"STATUS=stopped reason=budget_exceeded elapsed_h={elapsed_h:.4f}")
        return

    cur = state["current_run"]
    summary = run_status.inspect(
        cur["run_dir"], pid=cur.get("pid"), t=t,
    )
    verdict = summary["verdict"]
    print(f"VERDICT={verdict} cfg={cur['config_id']} iter={summary['last_iter']}/{t.expected_iterations} "
          f"best_score={summary['best_score']} best_combined={summary['best_combined']:.3f} "
          f"best_iter={summary['best_iter']} entropy={summary['entropy']:.4f} "
          f"recommendation=\"{summary['recommendation']}\"")

    if verdict == "RUNNING":
        append_journal(session_dir, {
            "event": "TICK", "config_id": cur["config_id"],
            "verdict": "RUNNING", "last_iter": summary["last_iter"],
            "best_iter": summary["best_iter"],
            "best_score": summary["best_score"],
            "best_combined": summary["best_combined"],
            "entropy": summary["entropy"], "kl": summary["approx_kl"],
            "clip_frac": summary["clip_frac"],
        })
        save_state(session_dir, state)
        print(f"ACTION=none elapsed_h={elapsed_h:.4f}")
        return

    if verdict == "DEPLOYABLE":
        if state["current_best"] is None:
            update_current_best(session_dir, cur["run_dir"], summary, state)
            state["deployable_history"].append({
                "ts": now, "config_id": cur["config_id"],
                "first_iter": summary["best_iter"],
                "combined": summary["best_combined"],
            })
            append_journal(session_dir, {
                "event": "DEPLOYABLE", "config_id": cur["config_id"],
                "best_iter": summary["best_iter"],
                "best_combined": summary["best_combined"],
                "per_n": summary["per_n"],
            })
            print(f"ACTION=snapshot_first_deployable iter={summary['best_iter']}")
        else:
            cur_combined = state["current_best"]["combined"] or 0.0
            if summary["best_combined"] > cur_combined + 0.02:
                update_current_best(session_dir, cur["run_dir"], summary, state)
                append_journal(session_dir, {
                    "event": "IMPROVED", "config_id": cur["config_id"],
                    "old_combined": cur_combined,
                    "new_combined": summary["best_combined"],
                })
                print(f"ACTION=improved combined {cur_combined:.3f} -> {summary['best_combined']:.3f}")
            else:
                append_journal(session_dir, {
                    "event": "TICK", "config_id": cur["config_id"],
                    "verdict": "DEPLOYABLE",
                    "best_combined": summary["best_combined"],
                    "current_best_combined": cur_combined,
                    "note": "deployable but not an improvement",
                })
                print("ACTION=none (deployable but not improvement)")
        save_state(session_dir, state)
        print(f"elapsed_h={elapsed_h:.4f}")
        return

    if verdict in ("COLLAPSED", "STALLED", "DIVERGED"):
        kill_result = kill_run(cur["pid"])
        append_journal(session_dir, {
            "event": "KILL", "config_id": cur["config_id"],
            "pid": cur["pid"], "kill_result": kill_result,
            "verdict": verdict, "reason": summary["reason"],
            "last_iter": summary["last_iter"],
            "best_score": summary["best_score"],
            "best_combined": summary["best_combined"],
        })
        print(f"ACTION=killed pid={cur['pid']} ({kill_result}) reason={verdict}")
        _advance(session_dir, state, train_iterations=train_iterations,
                 train_num_envs=train_num_envs, train_max_steps=train_max_steps,
                 train_checkpoint_every=train_checkpoint_every,
                 train_eval_every=train_eval_every,
                 train_eval_episodes=train_eval_episodes,
                 train_eval_n_present_counts=train_eval_n_present_counts,
                 tag_prefix=tag_prefix)
        return

    if verdict in ("DONE", "FAILED"):
        # On DONE, snapshot best.pt if it improves the current best (or first deployable).
        if verdict == "DONE":
            if state["current_best"] is None and is_deployable(summary, t):
                update_current_best(session_dir, cur["run_dir"], summary, state)
                state["deployable_history"].append({
                    "ts": now, "config_id": cur["config_id"],
                    "first_iter": summary["best_iter"],
                    "combined": summary["best_combined"],
                })
                append_journal(session_dir, {
                    "event": "DEPLOYABLE", "config_id": cur["config_id"],
                    "via": "DONE", "best_combined": summary["best_combined"],
                })
            elif state["current_best"] is not None:
                cur_combined = state["current_best"]["combined"] or 0.0
                if summary["best_combined"] > cur_combined + 0.02:
                    update_current_best(session_dir, cur["run_dir"], summary, state)
                    append_journal(session_dir, {
                        "event": "IMPROVED", "config_id": cur["config_id"],
                        "via": "DONE",
                        "old_combined": cur_combined,
                        "new_combined": summary["best_combined"],
                    })
        append_journal(session_dir, {
            "event": verdict, "config_id": cur["config_id"],
            "last_iter": summary["last_iter"],
            "best_iter": summary["best_iter"],
            "best_score": summary["best_score"],
            "best_combined": summary["best_combined"],
            "per_n": summary["per_n"],
        })
        print(f"ACTION={verdict.lower()}")
        _advance(session_dir, state, train_iterations=train_iterations,
                 train_num_envs=train_num_envs, train_max_steps=train_max_steps,
                 train_checkpoint_every=train_checkpoint_every,
                 train_eval_every=train_eval_every,
                 train_eval_episodes=train_eval_episodes,
                 train_eval_n_present_counts=train_eval_n_present_counts,
                 tag_prefix=tag_prefix)
        return

    print(f"ACTION=unknown verdict={verdict}")


def _advance(session_dir, state, *, train_iterations, train_num_envs,
             train_max_steps, train_checkpoint_every, train_eval_every,
             train_eval_episodes, train_eval_n_present_counts, tag_prefix):
    state_path = os.path.join(session_dir, "state.json")
    save_state(session_dir, state)  # persist the prior-run handling first
    nxt = hp_search.next_config(state_path)
    if nxt is None:
        end_session(session_dir, state,
                    reason="search space exhausted",
                    event="SEARCH_EXHAUSTED")
        print("ACTION=stop reason=search_exhausted")
        return
    cid, cfg = nxt
    pid, run_dir = start_run(
        session_dir=session_dir, config_id=cid, config=cfg,
        tag_prefix=tag_prefix, iterations=train_iterations,
        num_envs=train_num_envs, max_steps=train_max_steps,
        checkpoint_every=train_checkpoint_every,
        eval_every=train_eval_every,
        eval_episodes=train_eval_episodes,
        eval_n_present_counts=train_eval_n_present_counts,
    )
    state = load_state(session_dir)  # next_config rewrote state
    state["tried_config_ids"] = list(set(state["tried_config_ids"] + [cid]))
    state["current_run"] = {
        "config_id": cid, "config": cfg, "pid": pid,
        "run_dir": run_dir, "started_ts": time.time(),
        "log_path": os.path.join(session_dir, f"{cid}.log"),
    }
    save_state(session_dir, state)
    append_journal(session_dir, {
        "event": "START", "config_id": cid, "config": cfg,
        "pid": pid, "run_dir": run_dir,
    })
    print(f"ACTION=advanced cfg={cid} pid={pid} run_dir={run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--train-iterations", type=int, default=1500)
    ap.add_argument("--train-num-envs", type=int, default=8)
    ap.add_argument("--train-max-steps", type=int, default=600)
    ap.add_argument("--train-checkpoint-every", type=int, default=25)
    ap.add_argument("--train-eval-every", type=int, default=25)
    ap.add_argument("--train-eval-episodes", type=int, default=5)
    ap.add_argument("--train-eval-n-present-counts", type=str, default="1,4,7,10")
    ap.add_argument("--tag-prefix", type=str, default="auto-research")
    ap.add_argument("--warmup-iters", type=int, default=200)
    ap.add_argument("--deploy-min-succ", type=float, default=0.6)
    ap.add_argument("--deploy-max-form-err", type=float, default=0.30)
    ap.add_argument("--collapsed-no-new-best-iters", type=int, default=150)
    ap.add_argument("--stalled-window-iters", type=int, default=250)
    args = ap.parse_args()

    t = run_status.Thresholds(
        warmup_iters=args.warmup_iters,
        deploy_min_succ=args.deploy_min_succ,
        deploy_max_form_err=args.deploy_max_form_err,
        expected_iterations=args.train_iterations,
        collapsed_no_new_best_iters=args.collapsed_no_new_best_iters,
        stalled_window_iters=args.stalled_window_iters,
    )
    tick(
        args.session_dir,
        train_iterations=args.train_iterations,
        train_num_envs=args.train_num_envs,
        train_max_steps=args.train_max_steps,
        train_checkpoint_every=args.train_checkpoint_every,
        train_eval_every=args.train_eval_every,
        train_eval_episodes=args.train_eval_episodes,
        train_eval_n_present_counts=args.train_eval_n_present_counts,
        tag_prefix=args.tag_prefix,
        t=t,
    )


if __name__ == "__main__":
    main()
