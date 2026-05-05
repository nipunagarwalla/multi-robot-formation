"""Tests for RandomTeleop's regime coverage and concurrent-grab behavior."""
from __future__ import annotations

import numpy as np
import pytest

from teleop import RandomTeleop
from fake_env import FakeHallwayEnv


def _active_count(env, e: int) -> int:
    return int((env.teleop_mask[e] < 0.5).sum())


def test_initial_regime_distribution_covers_all_active_counts():
    """At reset, the active count should span {1, 2, 3, 4} over many resets."""
    env = FakeHallwayEnv(num_envs=1)
    env.vector_reset()
    rt = RandomTeleop(
        env,
        init_regime_dist=[0.25, 0.25, 0.25, 0.25],
        max_concurrent_grabs=3,
        seed=7,
    )
    seen = set()
    for _ in range(200):
        env.vector_reset()
        rt.reset_env(0)
        seen.add(_active_count(env, 0))
        if seen == {1, 2, 3, 4}:
            break
    assert seen == {1, 2, 3, 4}, f"missing regimes: {set(range(1,5)) - seen}"


def test_multi_grab_can_hold_more_than_one_robot_concurrently():
    """With p_grab high and max_concurrent_grabs=3, multiple robots get grabbed."""
    env = FakeHallwayEnv(num_envs=1)
    env.vector_reset()
    rt = RandomTeleop(
        env,
        p_grab=0.5,
        p_release=0.0,
        max_concurrent_grabs=3,
        init_regime_dist=[0.0, 0.0, 0.0, 1.0],  # start with 4 active
        seed=11,
    )
    rt.reset_env(0)
    assert _active_count(env, 0) == 4

    max_grabs_seen = 0
    for _ in range(50):
        rt.step()
        env.vector_step(np.zeros((1, env.cfg["n_agents"], 2)))
        max_grabs_seen = max(max_grabs_seen, len(rt.grabs[0]))
    assert max_grabs_seen >= 2, f"only {max_grabs_seen} concurrent grabs ever observed"


def test_concurrent_grabs_capped_at_n_agents_minus_one():
    """At least one robot must remain active even when the user requests more grabs."""
    env = FakeHallwayEnv(num_envs=1)
    rt = RandomTeleop(
        env,
        p_grab=1.0,
        p_release=0.0,
        max_concurrent_grabs=99,  # absurd: should clamp internally
        init_regime_dist=[1.0, 0.0, 0.0, 0.0],  # start with 1 active = 3 grabs
        seed=3,
    )
    env.vector_reset()
    rt.reset_env(0)
    # immediately 3 grabbed, 1 active
    assert _active_count(env, 0) == 1
    for _ in range(20):
        rt.step()
        env.vector_step(np.zeros((1, env.cfg["n_agents"], 2)))
        assert _active_count(env, 0) >= 1, "all robots got teleop'd — cap broken"


def test_regime_coverage_during_rollout():
    """Over a long rollout, all four active counts should be observed at some step."""
    env = FakeHallwayEnv(num_envs=4)
    env.vector_reset()
    rt = RandomTeleop(
        env,
        p_grab=0.02,
        p_release=0.02,
        max_concurrent_grabs=3,
        init_regime_dist=[0.25, 0.25, 0.25, 0.25],
        seed=5,
    )
    for e in range(env.cfg["num_envs"]):
        rt.reset_env(e)

    seen = set()
    for t in range(2000):
        rt.step()
        env.vector_step(np.zeros((env.cfg["num_envs"], env.cfg["n_agents"], 2)))
        for e in range(env.cfg["num_envs"]):
            seen.add(_active_count(env, e))
        if t % 200 == 199:
            # periodically reset one env to refresh initial regimes
            ridx = t % env.cfg["num_envs"]
            env.reset_at(ridx)
            rt.reset_env(ridx)
        if seen == {1, 2, 3, 4}:
            return
    assert seen == {1, 2, 3, 4}, f"missing regimes after 2000 steps: {set(range(1,5)) - seen}"


def test_invalid_init_regime_dist_raises():
    env = FakeHallwayEnv(num_envs=1)
    with pytest.raises(ValueError):
        RandomTeleop(env, init_regime_dist=[0.5, 0.5])  # wrong length
    with pytest.raises(ValueError):
        RandomTeleop(env, init_regime_dist=[0.0, 0.0, 0.0, 0.0])  # all zero
