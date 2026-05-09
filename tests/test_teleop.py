"""Tests for spawn/delete + RandomTeleop dynamic-count behavior."""
from __future__ import annotations

import numpy as np
import pytest

from contract import MAX_AGENTS, MIN_AGENTS
from env_hallway import FormationHallwayEnv
from teleop import RandomTeleop


# --- env spawn / delete invariants ------------------------------------------

def test_initial_n_present_matches_config():
    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": 6})
    env.vector_reset()
    assert env.n_present(0) == 6


def test_spawn_increments_present_count():
    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": 4})
    env.vector_reset()
    before = env.n_present(0)
    idx = env.spawn(0)
    assert idx is not None
    assert env.n_present(0) == before + 1
    assert float(env.present_mask[0, idx]) > 0.5


def test_spawn_at_max_returns_none():
    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": MAX_AGENTS})
    env.vector_reset()
    assert env.n_present(0) == MAX_AGENTS
    assert env.spawn(0) is None
    assert env.n_present(0) == MAX_AGENTS


def test_delete_decrements_present_count():
    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": 4})
    env.vector_reset()
    assert env.delete(0, 0) is True
    assert env.n_present(0) == 3
    assert float(env.present_mask[0, 0]) < 0.5


def test_delete_clears_teleop():
    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": 4})
    env.vector_reset()
    env.set_teleop(0, 1, True)
    env.set_teleop_action(0, 1, np.array([0.5, 0.0]))
    assert float(env.teleop_mask[0, 1]) > 0.5
    env.delete(0, 1)
    assert float(env.teleop_mask[0, 1]) < 0.5
    assert env.teleop_vels[0, 1].abs().sum().item() == 0.0


def test_min_one_robot_invariant():
    """delete must refuse to take n_present below MIN_AGENTS."""
    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": MIN_AGENTS})
    env.vector_reset()
    assert env.n_present(0) == MIN_AGENTS
    # try to delete the lone present robot
    present_idx = next(
        i for i in range(MAX_AGENTS) if float(env.present_mask[0, i]) > 0.5
    )
    assert env.delete(0, present_idx) is False
    assert env.n_present(0) == MIN_AGENTS


def test_max_ten_robots_invariant():
    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": 4})
    env.vector_reset()
    while env.spawn(0) is not None:
        pass
    assert env.n_present(0) == MAX_AGENTS


def test_delete_then_spawn_reuses_slot():
    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": 4})
    env.vector_reset()
    env.delete(0, 1)
    assert env.spawn(0) == 1  # lowest free index


# --- RandomTeleop ------------------------------------------------------------

def test_random_teleop_visits_full_n_present_range():
    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": 4})
    env.vector_reset()
    rt = RandomTeleop(
        env, p_grab=0.0, p_release=0.0, p_spawn=0.5, p_delete=0.5,
        init_n_present_dist=[1.0] * MAX_AGENTS, seed=2,
    )
    rt.reset_env(0)
    seen = {env.n_present(0)}
    for _ in range(800):
        rt.step()
        env.vector_step(np.zeros((1, MAX_AGENTS, 2)))
        seen.add(env.n_present(0))
    # with high spawn/delete probs and 800 steps, should see most of the range
    assert MIN_AGENTS in seen, f"never saw n_present={MIN_AGENTS}; seen={seen}"
    assert MAX_AGENTS in seen, f"never saw n_present={MAX_AGENTS}; seen={seen}"


def test_random_teleop_initial_dist_pins_episode_count():
    """A one-hot init dist should pin the post-reset n_present to that value."""
    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": 4})
    env.vector_reset()
    dist = [0.0] * MAX_AGENTS
    dist[6] = 1.0  # n_present = 7
    rt = RandomTeleop(env, init_n_present_dist=dist, seed=4)
    for _ in range(20):
        env.reset_at(0)
        rt.reset_env(0)
        assert env.n_present(0) == 7


def test_random_teleop_holds_min_active_invariant():
    """At least one robot must remain active even with maxed-out grab pressure."""
    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": 4})
    env.vector_reset()
    rt = RandomTeleop(
        env, p_grab=1.0, p_release=0.0, p_spawn=0.0, p_delete=0.0,
        max_concurrent_grabs=99,  # absurd: should clamp internally
        init_n_present_dist=[0.0] * (MAX_AGENTS - 1) + [1.0],  # start with 10
        seed=3,
    )
    rt.reset_env(0)
    for _ in range(20):
        rt.step()
        env.vector_step(np.zeros((1, MAX_AGENTS, 2)))
        active = int((env.present_mask[0] > 0.5).logical_and(env.teleop_mask[0] < 0.5).sum())
        assert active >= 1, "all robots got teleop'd — cap broken"


def test_invalid_init_dist_raises():
    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": 4})
    with pytest.raises(ValueError):
        RandomTeleop(env, init_n_present_dist=[0.5, 0.5])  # wrong length
    with pytest.raises(ValueError):
        RandomTeleop(env, init_n_present_dist=[0.0] * MAX_AGENTS)  # all zero
