"""Rendered policy visualization for a trained FormationHallway checkpoint.

This script is intentionally separate from `eval_hallway.py` so you can run a
clean pygame-only visualization pass after training.

Usage:
  python code/eval_hallway_viz.py --weights runs/<ts>/weights/latest.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pygame
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoint import load_checkpoint
from env_hallway import FormationHallwayEnv
from model import Agent
from render_hallway import HallwayRenderer


def _build_agent(env, device):
    cfg = {
        "model": {
            "custom_model_config": {
                "activation": "relu",
                "msg_features": 32,
                "comm_range": 2.0,
                "use_masks": True,
            }
        }
    }
    return Agent(env, cfg).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to a .pt checkpoint")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    env = FormationHallwayEnv(
        {
            "num_envs": 1,
            "max_time_steps": args.max_steps,
            "device": args.device,
            "render": False,
        }
    )
    agent = _build_agent(env, device)
    ckpt = load_checkpoint(args.weights, device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    renderer = HallwayRenderer()
    renderer.init()
    clock = pygame.time.Clock()

    obs = env.vector_reset()
    done_episodes = 0
    episode_reward = 0.0

    while done_episodes < args.episodes:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                renderer.close()
                return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                renderer.close()
                return

        with torch.no_grad():
            x = agent.format_input(obs, device)
            action, _, _, _ = agent.get_action_and_value(x)

        obs, r, done, info = env.vector_step(action.cpu().numpy())
        episode_reward += float(r[0])

        renderer.render(
            env,
            env_idx=0,
            episode_step=int(env.timesteps[0].item()),
            total_reward=episode_reward,
        )
        clock.tick(int(1 / env.cfg["dt"]))

        if done[0]:
            done_episodes += 1
            print(
                f"[viz] episode={done_episodes}/{args.episodes} "
                f"reached_goal={info[0]['goal_reached']} "
                f"reward={episode_reward:+.2f}"
            )
            env.reset_at(0)
            episode_reward = 0.0

    renderer.close()


if __name__ == "__main__":
    main()
