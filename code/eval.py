import os

# Important: make sure pygame is NOT headless
os.environ.pop("SDL_VIDEODRIVER", None)

import random

import numpy as np
import pygame
import torch
from env_line import PassageEnv
from model import Agent

num_agents = 5

agent_formation = (
    np.array(
        [
            [-0.5, 0],
            [0, 0],
            [1, 0],
            [1.5, 0],
            [2, 0],
        ]
    )
    * 0.5
).tolist()

config = {
    "seed": 0,
    "lr": 5e-5,
    "gamma": 0.995,
    "lambda": 0.95,
    "clip_param": 0.2,
    "entropy_coeff": 0.001,
    "vf_clip_param": 1.0,
    "vf_loss_coeff": 1.0,
    "max_grad_norm": 0.5,
    "norm_adv": True,
    "clip_vloss": True,
    "model": {
        "custom_model_config": {
            "activation": "relu",
            "msg_features": 32,
            "comm_range": 2.0,
        },
    },
    "env_config": {
        "world_dim": (4.0, 5.0),
        "dt": 0.05,
        "num_envs": 1,
        "device": "cpu",
        "n_agents": num_agents,
        "agent_formation": agent_formation,
        "placement_keepout_border": 1.0,
        "placement_keepout_wall": 1.5,
        "pos_noise_std": 0.0,
        "max_time_steps": 750,
        "communication_range": 20.0,
        "wall_width": 5.0,
        "gap_length": 2.3,
        "grid_px_per_m": 40,
        "agent_radius": 0.13,
        "render": True,
        "render_px_per_m": 160,
        "max_v": 1.0,
        "max_a": 1.0,
        "min_a": -1.0,
    },
}

random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

device = "cpu"

env = PassageEnv(config["env_config"])
agent = Agent(env, config).to(device)

weights_path = "weights/real-line2/weights_epoch1.pt"  # change epoch here
agent.load_state_dict(torch.load(weights_path, map_location=device))
agent.eval()

obs = env.vector_reset()
clock = pygame.time.Clock()

for step in range(env.cfg["max_time_steps"]):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            raise SystemExit

    with torch.no_grad():
        x = agent.format_input(obs, device)
        action, _, _, _ = agent.get_action_and_value(x)

    obs, reward, done, info = env.vector_step(action.cpu().numpy())

    env.render_ours(mode="human")
    clock.tick(int(1 / env.cfg["dt"]))

    if done[0]:
        print("done at step:", step)
        break

env.close()
