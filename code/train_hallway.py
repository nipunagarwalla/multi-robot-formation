"""PPO trainer for FormationHallwayEnv with random-teleop disturbance.

Mirrors code/train.py but adapts for:
  - dynamic active-cluster size (4 -> square, 3 -> tri, 2 -> line)
  - teleop_mask + present_mask in the observation
  - loss masking so gradients flow only through policy-controlled robots
  - persistent metrics via RunLogger (config.json + iterations.csv + episodes.jsonl)

Run:
  python code/train_hallway.py --iterations 200 --tag hallway-v0
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoint import load_checkpoint, save_checkpoint
from contract import REWARD_COEFFS
from device_utils import pick_device
from env_hallway import FormationHallwayEnv
from metrics import EpisodeAccumulator, RunLogger
from model import Agent
from teleop import RandomTeleop


def _parse_float_list(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return vals


def _parse_int_list(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected comma-separated ints")
    return vals


def _apply_fixed_active_count(env: FormationHallwayEnv, env_idx: int, active_count: int):
    for r in range(env.cfg["n_agents"]):
        is_teleop = r >= active_count
        env.set_teleop(env_idx, r, is_teleop)
        if is_teleop:
            env.set_teleop_action(env_idx, r, np.zeros(2, dtype=np.float32))


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _score_eval(per_active: dict[int, dict]) -> float:
    """Worst-regime-first score for selecting a stable 4/3/2 checkpoint."""
    if not per_active:
        return float("-inf")
    successes = [float(v["success_rate"]) for v in per_active.values()]
    fwd = [float(v["mean_forward_velocity"]) for v in per_active.values()]
    forms = [float(v["mean_formation_error"]) for v in per_active.values()]
    collisions = [float(v.get("mean_collisions", 0.0)) for v in per_active.values()]
    wall_hits = [float(v.get("mean_wall_hits", 0.0)) for v in per_active.values()]
    wall_contacts = [float(v.get("mean_wall_contact_steps", 0.0)) for v in per_active.values()]
    backward = [float(v.get("mean_backward_steps", 0.0)) for v in per_active.values()]
    min_wall_margins = [float(v.get("mean_min_wall_margin", 1.0)) for v in per_active.values()]
    wall_margin_shortfall = max(0.0, 0.12 - min(min_wall_margins))
    return float(
        1000.0 * min(successes)
        + 250.0 * float(np.mean(successes))
        + 25.0 * float(np.mean(fwd))
        - 50.0 * max(forms)
        - 0.5 * float(np.mean(collisions))
        - 5.0 * float(np.mean(wall_hits))
        - 3.0 * float(np.mean(wall_contacts))
        - 10.0 * float(np.mean(backward))
        - 100.0 * wall_margin_shortfall
    )


def _run_fixed_regime_eval(
    agent: Agent,
    config: dict,
    device: torch.device,
    active_counts: list[int],
    episodes: int,
    max_steps: int,
    seed: int,
) -> dict:
    env_cfg = dict(config["env_config"])
    env_cfg.update({"num_envs": 1, "max_time_steps": max_steps, "render": False})
    eval_env = FormationHallwayEnv(env_cfg)
    previous_training_state = agent.training
    agent.eval()

    per_active: dict[int, dict] = {}
    with torch.no_grad():
        for active_count in active_counts:
            records = []
            obs = eval_env.vector_reset()
            _apply_fixed_active_count(eval_env, 0, active_count)
            obs = eval_env.get_obs_tensor()
            acc = EpisodeAccumulator(eval_env.cfg["n_agents"])
            eps_done = 0
            while eps_done < episodes:
                x = agent.format_input(obs, device)
                action, _, _, _ = agent.get_action_and_value(x)
                obs, _r, done, infos = eval_env.vector_step(action, return_tensor_obs=True)
                info = infos[0]
                acc.update(
                    per_agent_rewards=[info["rewards"][k] for k in range(eval_env.cfg["n_agents"])],
                    active_count=int(info["active_count"]),
                    teleop_mask=obs["teleop_mask"][0].detach().cpu().tolist(),
                    formation_err=info["formation_error"],
                    fwd_velocity=float(info["fwd_velocity"]),
                    stalled=bool(info["stalled"]),
                    had_collision=bool(info["collided"]),
                    had_wall_hit=bool(info["wall_hit"]),
                    had_wall_contact=bool(info["wall_contact"]),
                    backward_step=bool(info["backward_step"]),
                    min_wall_margin=float(info["min_wall_margin"]),
                )
                if done[0]:
                    records.append(
                        acc.emit(
                            iteration=0,
                            env_id=0,
                            reached_goal=bool(info["goal_reached"]),
                        )
                    )
                    eps_done += 1
                    acc.reset()
                    eval_env.reset_at(0)
                    _apply_fixed_active_count(eval_env, 0, active_count)
                    obs = eval_env.get_obs_tensor()

            per_active[active_count] = {
                "episodes": len(records),
                "success_rate": _mean_or_zero([float(r["reached_goal"]) for r in records]),
                "mean_total_reward": _mean_or_zero([float(r["total_reward"]) for r in records]),
                "mean_episode_length": _mean_or_zero([float(r["episode_length"]) for r in records]),
                "mean_forward_velocity": _mean_or_zero(
                    [float(r["forward_velocity_mean"]) for r in records]
                ),
                "mean_formation_error": _mean_or_zero(
                    [float(r["formation_error_mean"]) for r in records]
                ),
                "mean_collisions": _mean_or_zero([float(r["num_collisions"]) for r in records]),
                "mean_wall_hits": _mean_or_zero([float(r["num_wall_hits"]) for r in records]),
                "mean_wall_contact_steps": _mean_or_zero(
                    [float(r.get("wall_contact_steps", 0.0)) for r in records]
                ),
                "mean_backward_steps": _mean_or_zero(
                    [float(r.get("backward_steps", 0.0)) for r in records]
                ),
                "mean_wall_margin": _mean_or_zero(
                    [float(r.get("mean_wall_margin", 0.0)) for r in records]
                ),
                "mean_min_wall_margin": _mean_or_zero(
                    [float(r.get("min_wall_margin", 0.0)) for r in records]
                ),
            }

    eval_env.close()
    if previous_training_state:
        agent.train()
    else:
        agent.eval()

    score = _score_eval(per_active)
    return {
        "seed": seed,
        "active_counts": active_counts,
        "episodes_per_active_count": episodes,
        "max_steps": max_steps,
        "score": score,
        "per_active": {str(k): v for k, v in per_active.items()},
    }


def make_config(num_envs: int, max_time_steps: int) -> dict:
    return {
        "seed": 0,
        "clip_param": 0.2,
        "entropy_coeff": 0.001,
        "vf_clip_param": 1.0,
        "vf_loss_coeff": 1.0,
        "max_grad_norm": 0.5,
        "norm_adv": True,
        "clip_vloss": True,
        "num_sgd_iter": 8,
        "lr": 5e-5,
        "gamma": 0.995,
        "lambda": 0.95,
        "model": {
            "custom_model_config": {
                "activation": "relu",
                "msg_features": 32,
                "comm_range": 2.0,
                "use_masks": True,
                "mask_teleop_edges": True,
            },
        },
        "env_config": {
            "num_envs": num_envs,
            "device": "cpu",
            "max_time_steps": max_time_steps,
            "render": False,
        },
        "teleop": {
            "p_grab": 0.005,
            "p_release": 0.01,
            "drift_speed": 0.6,
            "max_concurrent_grabs": 3,
            "init_regime_dist": [0.05, 0.35, 0.35, 0.25],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--tag", type=str, default="hallway-v0")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--minibatch-steps", type=int, default=64)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument(
        "--no-teleop",
        action="store_true",
        help="disable random-teleop disturbance (debug only)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda | mps. 'auto' picks cuda > mps > cpu.",
    )
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to a .pt checkpoint to resume from "
        "(continues iteration numbering, restores optimizer state)",
    )
    ap.add_argument("--p-grab", type=float, default=None)
    ap.add_argument("--p-release", type=float, default=None)
    ap.add_argument("--teleop-drift-speed", type=float, default=None)
    ap.add_argument("--max-concurrent-grabs", type=int, default=None)
    ap.add_argument(
        "--init-regime-dist",
        type=_parse_float_list,
        default=None,
        help="comma-separated probabilities for active counts 1,2,3,4",
    )
    ap.add_argument(
        "--save-best-on-eval",
        action="store_true",
        help="periodically run fixed-regime eval and update weights/best.pt on improvement",
    )
    ap.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="eval cadence in training iterations when --save-best-on-eval is set",
    )
    ap.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="episodes per fixed active count for periodic best-checkpoint eval",
    )
    ap.add_argument(
        "--eval-max-steps",
        type=int,
        default=None,
        help="max steps for periodic eval; defaults to --max-steps",
    )
    ap.add_argument(
        "--eval-active-counts",
        type=_parse_int_list,
        default=[4, 3, 2],
        help="comma-separated active counts to score for best checkpoint selection",
    )
    args = ap.parse_args()

    config = make_config(num_envs=args.num_envs, max_time_steps=args.max_steps)
    if args.save_best_on_eval and args.eval_every <= 0:
        raise ValueError("--eval-every must be positive when --save-best-on-eval is set")
    bad_counts = [k for k in args.eval_active_counts if k < 1 or k > 4]
    if bad_counts:
        raise ValueError(f"--eval-active-counts values must be in [1,4], got {bad_counts}")
    if args.p_grab is not None:
        config["teleop"]["p_grab"] = args.p_grab
    if args.p_release is not None:
        config["teleop"]["p_release"] = args.p_release
    if args.teleop_drift_speed is not None:
        config["teleop"]["drift_speed"] = args.teleop_drift_speed
    if args.max_concurrent_grabs is not None:
        config["teleop"]["max_concurrent_grabs"] = args.max_concurrent_grabs
    if args.init_regime_dist is not None:
        config["teleop"]["init_regime_dist"] = args.init_regime_dist
    config["seed"] = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    device = pick_device(args.device)
    config["env_config"]["device"] = str(device)
    print(f"[device] using {device}")

    env = FormationHallwayEnv(config["env_config"])
    agent = Agent(env, config).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config["lr"], eps=1e-5)

    start_iteration = 0
    resume_meta = None
    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, device)
        # strict=False so checkpoints saved before comm_range was a buffer
        # still load. The buffer is deterministic from cfg so it's safe to
        # allowlist; everything else missing or unexpected still raises.
        missing, unexpected = agent.load_state_dict(ckpt["agent"], strict=False)
        benign = {"model.comm_range"}
        real_missing = [k for k in missing if k not in benign]
        if real_missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch  missing={real_missing}  unexpected={list(unexpected)}"
            )
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_iteration = int(ckpt["iteration"])
        resume_meta = {
            "from": os.path.abspath(args.resume),
            "from_iteration": start_iteration,
            "had_optimizer_state": ckpt["optimizer"] is not None,
        }
        print(
            f"[resume] from {resume_meta['from']} at iter={start_iteration}"
            + (
                ""
                if resume_meta["had_optimizer_state"]
                else "  (NO optimizer state — Adam moments restart)"
            )
        )

    teleop = (
        None
        if args.no_teleop
        else RandomTeleop(
            env,
            p_grab=config["teleop"]["p_grab"],
            p_release=config["teleop"]["p_release"],
            drift_speed=config["teleop"]["drift_speed"],
            max_concurrent_grabs=config["teleop"]["max_concurrent_grabs"],
            init_regime_dist=config["teleop"]["init_regime_dist"],
            seed=args.seed,
        )
    )

    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
    runs_dir = os.path.abspath(runs_dir)
    os.makedirs(runs_dir, exist_ok=True)
    logger = RunLogger(runs_dir, tag=args.tag)
    logger.write_config(
        {
            "ppo": {
                k: v
                for k, v in config.items()
                if k not in ("env_config", "model", "teleop")
            },
            "env": config["env_config"],
            "model": config["model"],
            "teleop": config["teleop"] if teleop is not None else None,
            "reward_coeffs": REWARD_COEFFS,
            "args": vars(args),
            "resume": resume_meta,
            "start_iteration": start_iteration,
        }
    )
    print(f"[run] {logger.run_id} -> {logger.dir}")

    # --- buffer allocation ---------------------------------------------
    nE = env.cfg["num_envs"]
    nA = env.cfg["n_agents"]
    T = env.cfg["max_time_steps"]
    actions_buf = torch.zeros((T, nE, nA, 2), device=device)
    logprobs_buf = torch.zeros((T, nE, nA), device=device)
    rewards_buf = torch.zeros((T, nE, nA), device=device)
    dones_buf = torch.zeros((T, nE), device=device)
    values_buf = torch.zeros((T, nE, nA), device=device)
    teleop_buf = torch.zeros((T, nE, nA), device=device)

    env.vector_reset()
    if teleop is not None:
        for e in range(nE):
            teleop.reset_env(e)
    next_obs = env.get_obs_tensor()
    next_done = torch.zeros(nE, device=device)
    accs = [EpisodeAccumulator(nA) for _ in range(nE)]

    global_step = 0
    start_time = time.time()
    obs_per_step: list[dict[str, torch.Tensor]] = []
    best_eval_score = float("-inf")
    best_eval_iteration = None

    iter_lo = start_iteration + 1
    iter_hi = start_iteration + args.iterations + 1
    for iteration in range(iter_lo, iter_hi):
        obs_per_step = []
        ep_rewards: list = []
        ep_lengths: list = []
        ep_reached_goal: list = []
        # accumulate per-active-count formation errors across all episodes
        # finished during this iteration
        ep_form_err_by_active: dict = {1: [], 2: [], 3: [], 4: []}

        for step in range(T):
            global_step += nE
            x = agent.format_input(next_obs, device)
            obs_per_step.append({k: v.detach().clone() for k, v in x.items()})
            dones_buf[step] = next_done
            teleop_buf[step] = x["teleop_mask"]

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(x)
                values_buf[step] = value
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            if teleop is not None:
                teleop.step()

            next_obs, _r_summed, done, infos = env.vector_step(
                action, return_tensor_obs=True
            )

            per_agent = torch.zeros(nE, nA, device=device)
            for e in range(nE):
                for k, rv in infos[e]["rewards"].items():
                    per_agent[e, k] = float(rv)
                accs[e].update(
                    per_agent_rewards=per_agent[e].tolist(),
                    active_count=int(infos[e]["active_count"]),
                    teleop_mask=next_obs["teleop_mask"][e].detach().cpu().tolist(),
                    formation_err=infos[e]["formation_error"],
                    fwd_velocity=float(infos[e]["fwd_velocity"]),
                    stalled=bool(infos[e]["stalled"]),
                    had_collision=bool(infos[e]["collided"]),
                    had_wall_hit=bool(infos[e]["wall_hit"]),
                    had_wall_contact=bool(infos[e]["wall_contact"]),
                    backward_step=bool(infos[e]["backward_step"]),
                    min_wall_margin=float(infos[e]["min_wall_margin"]),
                )
            rewards_buf[step] = per_agent
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

            for e, d in enumerate(done):
                if d:
                    ep_rewards.append(accs[e].total_reward)
                    ep_lengths.append(accs[e].length)
                    ep_reached_goal.append(bool(infos[e]["goal_reached"]))
                    for k, v in accs[e].formation_err_by_active.items():
                        if k in ep_form_err_by_active and v:
                            ep_form_err_by_active[k].extend(v)
                    logger.log_episode(
                        accs[e].emit(
                            iteration=iteration,
                            env_id=e,
                            reached_goal=bool(infos[e]["goal_reached"]),
                        )
                    )
                    accs[e].reset()
                    env.reset_at(e)
                    if teleop is not None:
                        teleop.reset_env(e)
            if any(done):
                next_obs = env.get_obs_tensor()

        # --- advantages ------------------------------------------------
        with torch.no_grad():
            next_value = agent.get_value(agent.format_input(next_obs, device)).to(
                device
            )
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                nextnonterminal = nextnonterminal.unsqueeze(-1)
                delta = (
                    rewards_buf[t]
                    + config["gamma"] * nextvalues * nextnonterminal
                    - values_buf[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + config["gamma"] * config["lambda"] * nextnonterminal * lastgaelam
                )
            returns_buf = advantages + values_buf

        # --- PPO update over flattened (time, env) minibatches --------
        obs_buf = {
            k: torch.stack([obs[k] for obs in obs_per_step], dim=0)
            for k in obs_per_step[0].keys()
        }
        flat_obs = {
            k: v.reshape(T * nE, *v.shape[2:])
            for k, v in obs_buf.items()
        }
        flat_actions = actions_buf.reshape(T * nE, nA, 2)
        flat_logprobs = logprobs_buf.reshape(T * nE, nA)
        flat_advantages = advantages.reshape(T * nE, nA)
        flat_returns = returns_buf.reshape(T * nE, nA)
        flat_values = values_buf.reshape(T * nE, nA)
        flat_teleop = teleop_buf.reshape(T * nE, nA)

        b_inds = np.arange(T * nE)
        minibatch_size = max(1, min(T * nE, args.minibatch_steps * nE))
        last_pg = last_vl = last_ent = last_kl = last_clip = last_grad = 0.0
        for epoch in tqdm(range(config["num_sgd_iter"]), leave=False):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), minibatch_size):
                mb_inds = torch.as_tensor(
                    b_inds[start : start + minibatch_size],
                    dtype=torch.long,
                    device=device,
                )
                mb_obs = {k: v.index_select(0, mb_inds) for k, v in flat_obs.items()}
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    mb_obs, flat_actions.index_select(0, mb_inds)
                )
                old_logprob = flat_logprobs.index_select(0, mb_inds)
                logratio = newlogprob - old_logprob
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clip_frac = (
                        ((ratio - 1.0).abs() > config["clip_param"])
                        .float()
                        .mean()
                        .item()
                    )

                mb_adv = flat_advantages.index_select(0, mb_inds)
                active = 1.0 - flat_teleop.index_select(0, mb_inds)
                if config["norm_adv"]:
                    active_adv = mb_adv[active > 0.5]
                    if active_adv.numel() > 1:
                        mb_adv = (mb_adv - active_adv.mean()) / (active_adv.std() + 1e-8)

                # Active-mask: only policy-controlled robots contribute to the loss
                norm = active.sum().clamp(min=1.0)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - config["clip_param"], 1 + config["clip_param"]
                )
                pg_loss = (torch.max(pg_loss1, pg_loss2) * active).sum() / norm

                mb_returns = flat_returns.index_select(0, mb_inds)
                if config["clip_vloss"]:
                    mb_values = flat_values.index_select(0, mb_inds)
                    v_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -config["vf_clip_param"],
                        config["vf_clip_param"],
                    )
                    v_clipped_loss = (v_clipped - mb_returns) ** 2
                    v_max = torch.max(v_unclipped, v_clipped_loss)
                    v_loss = 0.5 * (v_max * active).sum() / norm
                else:
                    v_loss = (
                        0.5
                        * (((newvalue - mb_returns) ** 2) * active).sum()
                        / norm
                    )

                ent_loss = (entropy * active).sum() / norm
                loss = (
                    pg_loss
                    - config["entropy_coeff"] * ent_loss
                    + v_loss * config["vf_loss_coeff"]
                )

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    agent.parameters(), config["max_grad_norm"]
                )
                optimizer.step()

                last_pg = pg_loss.item()
                last_vl = v_loss.item()
                last_ent = ent_loss.item()
                last_kl = approx_kl
                last_clip = clip_frac
                last_grad = float(
                    grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                )

        # Active-only per-iteration mean reward (teleop'd slots are zero)
        mean_reward_iter = float(rewards_buf.sum().item()) / max(T * nE, 1)
        mean_ep_len = float(np.mean(ep_lengths)) if ep_lengths else float("nan")
        success_rate = (
            float(np.mean(ep_reached_goal)) if ep_reached_goal else float("nan")
        )

        def _mean_or_nan(xs):
            return float(np.mean(xs)) if xs else float("nan")

        per_active_mean = {
            k: _mean_or_nan(ep_form_err_by_active.get(k, []))
            for k in (1, 2, 3, 4)
        }
        per_active_n = {k: len(ep_form_err_by_active.get(k, [])) for k in (1, 2, 3, 4)}

        wall = time.time() - start_time
        logger.log_iter(
            iter=iteration,
            wall_time_s=round(wall, 2),
            env_steps=global_step,
            policy_loss=last_pg,
            value_loss=last_vl,
            entropy=last_ent,
            total_loss=last_pg
            + config["vf_loss_coeff"] * last_vl
            - config["entropy_coeff"] * last_ent,
            approx_kl=last_kl,
            clip_frac=last_clip,
            grad_norm=last_grad,
            mean_reward=mean_reward_iter,
            mean_episode_length=mean_ep_len,
            lr=config["lr"],
            success_rate=success_rate,
            formation_error_active_1=per_active_mean[1],
            formation_error_active_2=per_active_mean[2],
            formation_error_active_3=per_active_mean[3],
            formation_error_active_4=per_active_mean[4],
            n_episodes_active_1=per_active_n[1],
            n_episodes_active_2=per_active_n[2],
            n_episodes_active_3=per_active_n[3],
            n_episodes_active_4=per_active_n[4],
        )
        print(
            f"iter {iteration:4d}  rew {mean_reward_iter:+.4f}  "
            f"succ {success_rate*100:5.1f}%  "
            f"pg {last_pg:+.4f}  v {last_vl:.4f}  ent {last_ent:+.3f}  "
            f"kl {last_kl:+.4f}  ep_len {mean_ep_len:.1f}  "
            f"form[1/2/3/4]="
            f"{per_active_mean[1]:.3f}/{per_active_mean[2]:.3f}/"
            f"{per_active_mean[3]:.3f}/{per_active_mean[4]:.3f}"
        )

        if iteration % args.checkpoint_every == 0 or iteration == iter_hi - 1:
            ckpt = logger.checkpoint_path(iteration)
            save_checkpoint(ckpt, agent, optimizer, iteration)
            logger.update_latest_symlink(ckpt)

        if args.save_best_on_eval and (
            iteration % args.eval_every == 0 or iteration == iter_hi - 1
        ):
            eval_summary = _run_fixed_regime_eval(
                agent=agent,
                config=config,
                device=device,
                active_counts=args.eval_active_counts,
                episodes=args.eval_episodes,
                max_steps=args.eval_max_steps or args.max_steps,
                seed=args.seed + iteration,
            )
            eval_summary["iter"] = iteration
            eval_summary["best_so_far"] = False
            ckpt = logger.checkpoint_path(iteration)
            if not os.path.exists(ckpt):
                save_checkpoint(ckpt, agent, optimizer, iteration)
            score = float(eval_summary["score"])
            if score > best_eval_score:
                best_eval_score = score
                best_eval_iteration = iteration
                eval_summary["best_so_far"] = True
                logger.update_best_symlink(ckpt)
                logger.write_best_eval(eval_summary)
            logger.log_eval(eval_summary)
            per_active = eval_summary["per_active"]
            regime_bits = []
            for k in args.eval_active_counts:
                row = per_active[str(k)]
                regime_bits.append(
                    f"{k}:succ={row['success_rate']*100:.0f}%"
                    f",form={row['mean_formation_error']:.3f}"
                    f",vy={row['mean_forward_velocity']:+.2f}"
                    f",wall={row['mean_wall_contact_steps']:.1f}"
                    f",back={row['mean_backward_steps']:.1f}"
                )
            print(
                f"[eval] iter {iteration}  score={score:.2f}  "
                f"best={best_eval_score:.2f}@{best_eval_iteration}  "
                + "  ".join(regime_bits)
            )

    logger.close()
    print(f"[done] last ckpt: {logger.checkpoint_path(iter_hi - 1)}")


if __name__ == "__main__":
    main()
