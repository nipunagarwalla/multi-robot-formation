"""Smoke tests for tensor obs/actions used by the faster trainer path."""
import torch

from env_hallway import FormationHallwayEnv
from model import Agent
from train_hallway import make_config


def test_tensor_obs_and_action_fastpath_shapes():
    env = FormationHallwayEnv({"num_envs": 2, "max_time_steps": 5})
    env.vector_reset()
    obs = env.get_obs_tensor()

    config = make_config(num_envs=2, max_time_steps=5)
    agent = Agent(env, config)
    x = agent.format_input(obs, torch.device("cpu"))

    assert x["pos"].shape == (2, env.cfg["n_agents"], 2)
    assert x["teleop_mask"].shape == (2, env.cfg["n_agents"])

    with torch.no_grad():
        action, _, _, value = agent.get_action_and_value(x)
    next_obs, rewards, dones, infos = env.vector_step(action, return_tensor_obs=True)

    assert next_obs["pos"].shape == (2, env.cfg["n_agents"], 2)
    assert len(rewards) == 2
    assert len(dones) == 2
    assert len(infos) == 2
    assert value.shape == (2, env.cfg["n_agents"])
