"""Unit tests for the circle target formation and env smoke."""
import math

import numpy as np
import pytest
import torch

from contract import CIRCLE_SIDE, MAX_AGENTS, MIN_AGENTS
from env_hallway import FormationHallwayEnv, target_formation_positions


def test_shapes():
    for n in range(1, MAX_AGENTS + 1):
        slots = target_formation_positions(n)
        assert slots.shape == (n, 2), f"n={n}: got {slots.shape}"


def test_centred_at_origin():
    for n in range(2, MAX_AGENTS + 1):
        slots = target_formation_positions(n)
        c = slots.mean(dim=0)
        assert torch.allclose(c, torch.zeros(2), atol=1e-5), f"n={n} centroid={c}"


def test_n1_returns_origin():
    slots = target_formation_positions(1)
    assert slots.shape == (1, 2)
    assert torch.allclose(slots, torch.zeros(1, 2), atol=1e-7)


def test_n2_is_horizontal_pair():
    slots = target_formation_positions(2, scale=1.0).numpy()
    # both at y=0, centred on origin in x, separation = scale
    assert abs(slots[0, 1]) < 1e-5 and abs(slots[1, 1]) < 1e-5
    assert abs(slots[1, 0] - slots[0, 0] - 1.0) < 1e-5


@pytest.mark.parametrize("n", list(range(3, MAX_AGENTS + 1)))
def test_circle_radius_scaling(n):
    """For n >= 3, every slot sits at distance r(n) = scale/(2 sin(pi/n))."""
    scale = 1.0
    slots = target_formation_positions(n, scale=scale).numpy()
    expected_r = scale / (2.0 * math.sin(math.pi / n))
    radii = np.linalg.norm(slots, axis=1)
    assert np.allclose(radii, expected_r, atol=1e-5), (
        f"n={n}: expected r={expected_r}, got radii={radii}"
    )


@pytest.mark.parametrize("n", list(range(3, MAX_AGENTS + 1)))
def test_neighbouring_chord_length(n):
    """Adjacent slots are exactly `scale` apart along the chord."""
    scale = 1.0
    slots = target_formation_positions(n, scale=scale).numpy()
    # slots are emitted in a consistent angular order, so consecutive pairs
    # plus the wrap-around pair are the chord-length neighbours
    chords = []
    for i in range(n):
        chords.append(np.linalg.norm(slots[i] - slots[(i + 1) % n]))
    assert np.allclose(chords, scale, atol=1e-5), (
        f"n={n}: expected all chords={scale}, got {chords}"
    )


def test_unsupported_count_raises():
    with pytest.raises(ValueError):
        target_formation_positions(0)


def test_circle_side_exceeds_robot_diameter():
    """Adjacent robots in formation must not overlap. CIRCLE_SIDE > 2*AGENT_RADIUS."""
    from contract import AGENT_RADIUS
    assert CIRCLE_SIDE > 2 * AGENT_RADIUS, (
        f"CIRCLE_SIDE={CIRCLE_SIDE} <= 2*AGENT_RADIUS={2*AGENT_RADIUS}"
    )


def test_env_step_shapes():
    env = FormationHallwayEnv({"num_envs": 2, "initial_agents": 4})
    obs = env.vector_reset()
    assert len(obs) == 2
    a = np.zeros((2, MAX_AGENTS, 2), dtype=np.float32)
    obs, r, done, info = env.vector_step(a)
    assert len(obs) == 2 and len(r) == 2 and len(done) == 2 and len(info) == 2
    assert "teleop_mask" in obs[0] and "present_mask" in obs[0]
    assert len(obs[0]["teleop_mask"]) == MAX_AGENTS
    assert len(obs[0]["present_mask"]) == MAX_AGENTS
    assert info[0]["n_present"] == 4
    assert info[0]["n_active"] == 4


def test_teleop_override_present_robot():
    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": 4})
    env.vector_reset()
    env.set_teleop(0, 2, True)
    env.set_teleop_action(0, 2, np.array([0.5, 0.0]))
    a = np.zeros((1, MAX_AGENTS, 2), dtype=np.float32)
    p_before = env.ps[0, 2].clone()
    env.vector_step(a)
    p_after = env.ps[0, 2].clone()
    assert (p_after - p_before).abs().sum().item() > 1e-4
    obs, r, done, info = env.vector_step(a)
    assert info[0]["rewards"][2] == 0.0


def test_teleop_release():
    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": 4})
    env.vector_reset()
    env.set_teleop(0, 1, True)
    env.set_teleop_action(0, 1, np.array([0.5, 0.5]))
    env.vector_step(np.zeros((1, MAX_AGENTS, 2)))
    assert env.teleop_mask[0, 1] == 1.0
    env.set_teleop(0, 1, False)
    assert env.teleop_mask[0, 1] == 0.0
    assert env.teleop_vels[0, 1].abs().sum().item() == 0.0


def test_teleop_on_non_present_is_noop():
    """set_teleop(active=True) on a non-present slot should NOT mark it teleop'd."""
    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": 2})
    env.vector_reset()
    # slot 5 is non-present (only 0, 1 are present)
    assert float(env.present_mask[0, 5]) < 0.5
    env.set_teleop(0, 5, True)
    assert float(env.teleop_mask[0, 5]) < 0.5
