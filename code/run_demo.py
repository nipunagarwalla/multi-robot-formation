"""Interactive demo: trained policy + keyboard teleop in a pygame window.

Press 1-4 to toggle teleop on each robot, WASD to drive the most-recently
selected one, 0 to release all. ESC or window-close to quit.

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
from device_utils import pick_device
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
                "comm_range": 2.0,
                "use_masks": True,
                "mask_teleop_edges": True,
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
    ap.add_argument("--device", type=str, default="auto",
                    help="auto | cpu | cuda | mps")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"[device] using {device}")

    env = FormationHallwayEnv(
        {"num_envs": 1, "max_time_steps": args.max_steps, "device": str(device)}
    )
    agent = _build_agent(env, device)
    ckpt = load_checkpoint(args.weights, device)
    # strict=False so older checkpoints (saved before comm_range was a buffer)
    # still load — the buffer is deterministic from cfg["comm_range"].
    missing, unexpected = agent.load_state_dict(ckpt["agent"], strict=False)
    benign = {"model.comm_range"}
    real_missing = [k for k in missing if k not in benign]
    if real_missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch  missing={real_missing}  unexpected={list(unexpected)}"
        )
    agent.eval()

    renderer = HallwayRenderer()
    renderer.init()
    teleop = KeyboardTeleop(env, env_idx=0)
    clock = pygame.time.Clock()

    obs = env.vector_reset()
    total_r = 0.0
    step = 0
    running = True
    print("[demo] keys: 1-4 toggle teleop · WASD drive · Z/X speed · 0 release · ESC quit")

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

        obs, r, done, info = env.vector_step(action)
        total_r += float(r[0])
        step += 1

        renderer.render(
            env,
            env_idx=0,
            episode_step=step,
            total_reward=total_r,
            teleop_speed=teleop.drive_speed,
        )
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
