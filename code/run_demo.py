"""Interactive demo: trained policy + keyboard teleop in a pygame window.

Up to 10 robots in an 8 m x 8 m square arena, always forming a circle and
heading north toward the goal line.

Keys:
  1-9        toggle teleop on robot 1-9
  0          toggle teleop on robot 10
  W A S D    drive the most-recently-selected teleop robot
  Z / X      decrease / increase teleop drive speed (0.25 m/s steps)
  =  / +     spawn a new robot (no-op at MAX_AGENTS)
  -  / _     delete the selected (or highest-index) robot (no-op at MIN_AGENTS=1)
  R          release all teleop'd robots
  ESC        quit

Usage:
  python code/run_demo.py --weights runs/<ts>/weights/latest.pt
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
from teleop import KeyboardTeleop


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
    ap.add_argument("--weights", required=True)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reset-on-done", action="store_true",
                    help="auto-reset the env when an episode ends")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    env = FormationHallwayEnv(
        {"num_envs": 1, "max_time_steps": args.max_steps, "device": "cpu"}
    )
    agent = _build_agent(env, device)
    ckpt = load_checkpoint(args.weights, device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    renderer = HallwayRenderer()
    renderer.init()
    teleop = KeyboardTeleop(env, env_idx=0)
    clock = pygame.time.Clock()

    obs = env.vector_reset()
    total_r = 0.0
    step = 0
    running = True
    print("[demo] keys: 1-9/0 toggle teleop · WASD drive · Z/X speed · "
          "=/- spawn/delete · R release · ESC quit")

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                break
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False
                break
            teleop.handle_event(ev)
        if not running:
            break

        teleop.apply()

        with torch.no_grad():
            x = agent.format_input(obs, device)
            action, _, _, _ = agent.get_action_and_value(x)

        obs, r, done, info = env.vector_step(action.cpu().numpy())
        total_r += float(r[0])
        step += 1

        renderer.render(env, env_idx=0, episode_step=step, total_reward=total_r,
                        drive_speed=teleop.drive_speed)
        clock.tick(int(1 / env.cfg["dt"]))

        if done[0]:
            print(f"[demo] episode end at step={step}  reached_goal={info[0]['goal_reached']}  R={total_r:+.2f}")
            if args.reset_on_done:
                env.reset_at(0)
                # forget any held teleop state on reset
                teleop.selected = None
                teleop.pressed.clear()
                total_r = 0.0
                step = 0
            else:
                running = False

    renderer.close()


if __name__ == "__main__":
    main()
