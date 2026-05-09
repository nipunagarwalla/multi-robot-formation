"""Headless or rendered evaluation of a trained FormationHallway policy.

Writes <run-dir>/eval.json with per-episode reward, forward velocity,
formation error, success bool, and the mean / min / max n_present visited
during the episode. With --teleop, RandomTeleop drives all four
mechanisms (grab / release / spawn / delete) so episodes span the full
1..MAX_AGENTS regime spectrum; the JSON includes a per_regime breakdown
bucketed by min_n_present.

Usage:
  python code/eval_hallway.py --weights runs/<ts>/weights/latest.pt --episodes 20
  python code/eval_hallway.py --weights runs/<ts>/weights/latest.pt --render
  python code/eval_hallway.py --weights runs/<ts>/weights/latest.pt --teleop
  python code/eval_hallway.py --weights runs/<ts>/weights/latest.pt --fixed-n-present 7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pygame
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoint import load_checkpoint
from contract import MAX_AGENTS, MIN_AGENTS
from env_hallway import FormationHallwayEnv
from metrics import EpisodeAccumulator
from model import Agent
from teleop import RandomTeleop


def _build_agent(env, device):
    cfg = {
        "model": {
            "custom_model_config": {
                "activation": "relu",
                "msg_features": 32,
                "comm_range": 4.0,
                "use_masks": True,
            }
        }
    }
    return Agent(env, cfg).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to a .pt checkpoint")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--teleop", action="store_true",
                    help="apply RandomTeleop disturbance during eval")
    ap.add_argument(
        "--fixed-n-present",
        type=int,
        default=None,
        help=f"hold n_present constant at this value ({MIN_AGENTS}..{MAX_AGENTS}); "
             "implicitly disables --teleop spawn/delete events",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="path for eval.json (defaults to alongside the weights)")
    args = ap.parse_args()

    if args.fixed_n_present is not None:
        if not (MIN_AGENTS <= args.fixed_n_present <= MAX_AGENTS):
            raise SystemExit(
                f"--fixed-n-present must be in [{MIN_AGENTS}, {MAX_AGENTS}]"
            )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    render = args.render and not args.no_render
    if not render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    device = torch.device("cpu")

    env_cfg = {
        "num_envs": args.num_envs,
        "max_time_steps": args.max_steps,
        "device": "cpu",
        "render": False,
    }
    if args.fixed_n_present is not None:
        env_cfg["initial_agents"] = int(args.fixed_n_present)
    env = FormationHallwayEnv(env_cfg)
    agent = _build_agent(env, device)
    ckpt = load_checkpoint(args.weights, device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    teleop = None
    if args.teleop:
        teleop_kwargs = dict(seed=args.seed)
        if args.fixed_n_present is not None:
            # pin RandomTeleop to this n_present: zero spawn/delete probs
            # and a one-hot init distribution
            dist = [0.0] * MAX_AGENTS
            dist[args.fixed_n_present - 1] = 1.0
            teleop_kwargs.update(
                p_spawn=0.0,
                p_delete=0.0,
                init_n_present_dist=dist,
            )
        teleop = RandomTeleop(env, **teleop_kwargs)

    renderer = None
    clock = None
    if render:
        from render_hallway import HallwayRenderer
        renderer = HallwayRenderer()
        renderer.init()
        clock = pygame.time.Clock()

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.weights)), "..", "eval.json"
    )
    out_path = os.path.abspath(out_path)

    records = []
    eps_done = 0
    obs = env.vector_reset()
    if teleop is not None:
        for e in range(env.cfg["num_envs"]):
            teleop.reset_env(e)
    accs = [EpisodeAccumulator(env.cfg["n_agents"]) for _ in range(env.cfg["num_envs"])]
    t0 = time.time()

    while eps_done < args.episodes:
        with torch.no_grad():
            x = agent.format_input(obs, device)
            action, _, _, _ = agent.get_action_and_value(x)
        if teleop is not None:
            teleop.step()
        obs, _r, done, infos = env.vector_step(action.cpu().numpy())
        for e in range(env.cfg["num_envs"]):
            accs[e].update(
                per_agent_rewards=[infos[e]["rewards"][k] for k in range(MAX_AGENTS)],
                active_count=int(infos[e]["active_count"]),
                teleop_mask=obs[e]["teleop_mask"],
                n_present=int(infos[e]["n_present"]),
                formation_err=infos[e]["formation_error"],
                circle_radius=float(infos[e].get("circle_radius", 0.0)),
                fwd_velocity=float(infos[e]["fwd_velocity"]),
                stalled=bool(infos[e]["stalled"]),
                had_collision=bool(infos[e]["collided"]),
                had_wall_hit=bool(infos[e]["wall_hit"]),
            )

        if render:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    if renderer:
                        renderer.close()
                    raise SystemExit
            renderer.render(env, env_idx=0,
                            episode_step=accs[0].length,
                            total_reward=accs[0].total_reward)
            clock.tick(int(1 / env.cfg["dt"]))

        for e, d in enumerate(done):
            if not d:
                continue
            rec = accs[e].emit(iteration=0, env_id=e,
                               reached_goal=bool(infos[e]["goal_reached"]))
            records.append(rec)
            eps_done += 1
            accs[e].reset()
            env.reset_at(e)
            if teleop is not None:
                teleop.reset_env(e)
            if eps_done >= args.episodes:
                break

    # Per-regime bucketing keyed on min_n_present (the hardest cluster size
    # the policy was forced to handle during the episode).
    per_regime: dict = {}
    for k in range(MIN_AGENTS, MAX_AGENTS + 1):
        bucket = [r for r in records if r.get("min_n_present") == k]
        if not bucket:
            per_regime[str(k)] = {"n_episodes": 0}
            continue
        per_regime[str(k)] = {
            "n_episodes": len(bucket),
            "success_rate": float(np.mean([r["reached_goal"] for r in bucket])),
            "mean_total_reward": float(np.mean([r["total_reward"] for r in bucket])),
            "mean_episode_length": float(np.mean([r["episode_length"] for r in bucket])),
            "mean_forward_velocity": float(np.mean([r["forward_velocity_mean"] for r in bucket])),
            "mean_formation_error": float(np.mean([r["formation_error_mean"] for r in bucket])),
            "mean_circle_radius": float(np.mean([r["circle_radius_mean"] for r in bucket])),
        }

    summary = {
        "weights": os.path.abspath(args.weights),
        "wall_time_s": round(time.time() - t0, 2),
        "episodes": len(records),
        "fixed_n_present": args.fixed_n_present,
        "success_rate": float(np.mean([r["reached_goal"] for r in records])) if records else 0.0,
        "mean_total_reward": float(np.mean([r["total_reward"] for r in records])) if records else 0.0,
        "mean_episode_length": float(np.mean([r["episode_length"] for r in records])) if records else 0.0,
        "mean_forward_velocity": float(np.mean([r["forward_velocity_mean"] for r in records])) if records else 0.0,
        "mean_formation_error": float(np.mean([r["formation_error_mean"] for r in records])) if records else 0.0,
        "per_regime": per_regime,
        "records": records,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[eval] {args.episodes} episodes  "
        f"success={summary['success_rate']*100:.0f}%  "
        f"mean_R={summary['mean_total_reward']:+.2f}  "
        f"mean_len={summary['mean_episode_length']:.1f}  "
        f"mean_v_y={summary['mean_forward_velocity']:+.3f}  "
        f"-> {out_path}"
    )
    if any(per_regime[str(k)].get("n_episodes", 0) > 0 for k in range(MIN_AGENTS, MAX_AGENTS + 1)):
        print("       per-regime (bucketed by min_n_present):")
        for k in range(MIN_AGENTS, MAX_AGENTS + 1):
            r = per_regime[str(k)]
            if r.get("n_episodes", 0) == 0:
                continue
            print(
                f"         n_present={k:2d}: n={r['n_episodes']:3d}  "
                f"succ={r['success_rate']*100:5.1f}%  "
                f"mean_v_y={r['mean_forward_velocity']:+.3f}  "
                f"form_err={r['mean_formation_error']:.3f}  "
                f"r_circ={r['mean_circle_radius']:.2f}"
            )
    if renderer:
        renderer.close()


if __name__ == "__main__":
    main()
