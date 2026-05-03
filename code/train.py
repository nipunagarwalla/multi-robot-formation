#%%

import os
import time
import numpy as np
import random
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from env_line import PassageEnv

from model import Agent


#%%

num_agents = 5
pentagon_coords = np.array([
    [np.cos(2 * np.pi * i / num_agents + np.pi/2 + np.pi/(num_agents+1)), \
     np.sin(2 * np.pi * i / num_agents + np.pi/2 + np.pi/(num_agents+1))]
    for i in range(num_agents)
])

scale_factor = 0.5
scaled_pentagon_coords = pentagon_coords * scale_factor

agent_formation = (np.array([[-0.5, 0],[0, 0], [1, 0],[1.5, 0], [2, 0]]) * 0.5).tolist()


config={
    "seed": 0,
    "framework": "torch",
    "env": "passage_env",
    "clip_param": 0.2,
    "entropy_coeff": 0.001,
    "train_batch_size": 65536,
    "sgd_minibatch_size": 4096,
    "vf_clip_param": 1.0,
    "vf_loss_coeff": 1.0,
    "max_grad_norm": 0.5,
    "norm_adv": True,
    "clip_vloss": True,
    "num_sgd_iter": 10,
    "num_gpus": 1,
    "num_envs_per_worker": 10,
    "lr": 5e-5,
    "gamma": 0.995,
    "lambda": 0.95,
    "batch_mode": "truncate_episodes",
    "observation_filter": "NoFilter",
    "model": {
        "custom_model": "model",
        "custom_action_dist": "hom_multi_action",
        "custom_model_config": {
            "activation": "relu",
            "msg_features": 32,
            "comm_range": 2.0,
        },
    },
    "env_config": {
        "world_dim": (4.0, 5.0),
        "dt": 0.05,
        "num_envs": 32,
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
        "render": False,
        "render_px_per_m": 160,
        "max_v": 1.0,
        "max_a": 1.0,
        "min_a": -1.0,
    },
    "render_env": False,
    "evaluation_interval": 1,
    "evaluation_num_episodes": 1,
    "evaluation_num_workers": 1,
    "evaluation_parallel_to_training": True,
    "evaluation_config": {
        "record_env": "videos",
        "render_env": True,
    },
}

#%%
random.seed(config['seed'])
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])
torch.backends.cudnn.deterministic = True

#%%
os.environ["SDL_VIDEODRIVER"]='dummy'
device = 'cpu'

env_config = config['env_config']
env = PassageEnv(env_config)

agent = Agent(env, config).to(device)
optimizer = optim.Adam(agent.parameters(), lr=config['lr'], eps=1e-5)


#%%
env.vector_reset()
returns = torch.zeros((env.cfg["num_envs"], env.cfg["n_agents"]))
selected_agent = 0
rew = 0


#%%

obs = list()
actions = torch.zeros((env.cfg["max_time_steps"], env.cfg["num_envs"]) + env.observation_space["pos"].shape).to(device)
logprobs = torch.zeros((env.cfg["max_time_steps"], env.cfg["num_envs"],env.cfg["n_agents"])).to(device)
rewards = torch.zeros((env.cfg["max_time_steps"], env.cfg["num_envs"],env.cfg["n_agents"])).to(device)
dones = torch.zeros((env.cfg["max_time_steps"], env.cfg["num_envs"])).to(device)
values = torch.zeros((env.cfg["max_time_steps"], env.cfg["num_envs"],env.cfg["n_agents"])).to(device)

global_step = 0
start_time = time.time()
next_obs = env.vector_reset()
next_done = torch.zeros(env.cfg['num_envs']).to(device)

num_iterations = config['train_batch_size']
batch_size = env.cfg["max_time_steps"]
minibatch_size = config['sgd_minibatch_size']

weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'real-line2')
os.makedirs(weights_dir, exist_ok=True)

for iteration in range(1, num_iterations + 1):
    obs = list()
    frames = []

    for step in range(0, env.cfg["max_time_steps"]):
        global_step += env.cfg['num_envs']
        obs.append(agent.format_input(next_obs, device))
        dones[step] = next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(agent.format_input(next_obs, device))
            values[step] = value
        actions[step] = action
        logprobs[step] = logprob

        next_obs, reward, done, infos = env.vector_step(action.cpu().numpy())
        next_done = np.array(done)

        returns = torch.zeros((env.cfg["num_envs"], env.cfg["n_agents"]))
        for idx in range(env.cfg["num_envs"]):
            info_instance = infos[idx]
            for key, agent_reward in info_instance["rewards"].items():
                returns[idx, key] += agent_reward
        rewards[step] = returns.to(device)
        next_done = torch.Tensor(next_done).to(device)

        for idx, done_env in enumerate(done):
            if done_env:
                env.reset_at(idx)

    print('mean rewards at iter {:4d}:'.format(iteration), torch.mean(rewards))


    with torch.no_grad():
        next_value = agent.get_value(agent.format_input(next_obs, device)).to(device)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(env.cfg["max_time_steps"])):
            if t == env.cfg["max_time_steps"] - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            nextnonterminal = nextnonterminal.unsqueeze(dim=-1)
            delta = rewards[t] + config['gamma'] * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + config['gamma'] * config['lambda'] * nextnonterminal * lastgaelam
        returns = advantages + values

    b_obs = obs
    b_logprobs = logprobs
    b_actions = actions
    b_advantages = advantages
    b_returns = returns
    b_values = values

    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in tqdm(range(config['num_sgd_iter'])):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size):
            mb_inds = b_inds[start]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > config['clip_param']).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if config['norm_adv']:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config['clip_param'], 1 + config['clip_param'])
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            if config['clip_vloss']:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -config['vf_clip_param'],
                    config['vf_clip_param'],
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - config['entropy_coeff'] * entropy_loss + v_loss * config['vf_loss_coeff']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config['max_grad_norm'])
            optimizer.step()
    print('loss:', loss.detach().cpu())
    torch.save(agent.state_dict(), os.path.join(weights_dir, f'weights_epoch{iteration}.pt'))

#%%
