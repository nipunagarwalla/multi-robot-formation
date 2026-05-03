"""Unit tests for the canonical formation slot positions and env smoke."""
import numpy as np
import pytest
import torch

from contract import FORMATION_SCALE, MAX_AGENTS
from env_hallway import FormationHallwayEnv, target_formation_positions


def test_shapes():
    for n in (1, 2, 3, 4):
        slots = target_formation_positions(n)
        assert slots.shape == (n, 2), f"n={n}: got {slots.shape}"


def test_centred_at_origin():
    for n in (2, 3, 4):
        slots = target_formation_positions(n)
        c = slots.mean(dim=0)
        assert torch.allclose(c, torch.zeros(2), atol=1e-5), f"n={n} centroid={c}"


def test_square_side_length():
    slots = target_formation_positions(4, scale=1.0).numpy()
    # Adjacent vertices: 4 sides each length 1.0
    sides = []
    for i in range(4):
        d = np.linalg.norm(slots[i] - slots[(i + 1) % 4])
        sides.append(d)
    sides.sort()
    # Two pairs of equal sides (square: 4 equal-length sides) — all 1.0
    assert all(abs(s - 1.0) < 1e-5 for s in sides), f"sides={sides}"


def test_triangle_equilateral():
    slots = target_formation_positions(3, scale=1.0).numpy()
    d01 = np.linalg.norm(slots[0] - slots[1])
    d12 = np.linalg.norm(slots[1] - slots[2])
    d02 = np.linalg.norm(slots[0] - slots[2])
    assert abs(d01 - d12) < 1e-5 and abs(d01 - d02) < 1e-5, f"sides=({d01},{d12},{d02})"
    assert abs(d01 - 1.0) < 1e-5


def test_line_horizontal():
    slots = target_formation_positions(2, scale=1.0).numpy()
    assert abs(slots[0, 1]) < 1e-5 and abs(slots[1, 1]) < 1e-5
    assert abs(slots[1, 0] - slots[0, 0] - 1.0) < 1e-5


def test_unsupported_count_raises():
    with pytest.raises(ValueError):
        target_formation_positions(5)


def test_env_step_shapes():
    env = FormationHallwayEnv({"num_envs": 2})
    obs = env.vector_reset()
    assert len(obs) == 2
    a = np.zeros((2, MAX_AGENTS, 2), dtype=np.float32)
    obs, r, done, info = env.vector_step(a)
    assert len(obs) == 2 and len(r) == 2 and len(done) == 2 and len(info) == 2
    assert "teleop_mask" in obs[0] and "present_mask" in obs[0]


def test_teleop_override():
    env = FormationHallwayEnv({"num_envs": 1})
    env.vector_reset()
    env.set_teleop(0, 2, True)
    env.set_teleop_action(0, 2, np.array([0.5, 0.0]))
    a = np.zeros((1, MAX_AGENTS, 2), dtype=np.float32)
    p_before = env.ps[0, 2].clone()
    env.vector_step(a)
    p_after = env.ps[0, 2].clone()
    # Teleop'd robot should have moved (override velocity), even with zero policy action
    assert (p_after - p_before).abs().sum().item() > 1e-4
    # Reward for teleop'd robot should be 0
    obs, r, done, info = env.vector_step(a)
    assert info[0]["rewards"][2] == 0.0


def test_teleop_release():
    env = FormationHallwayEnv({"num_envs": 1})
    env.vector_reset()
    env.set_teleop(0, 1, True)
    env.set_teleop_action(0, 1, np.array([0.5, 0.5]))
    env.vector_step(np.zeros((1, MAX_AGENTS, 2)))
    assert env.teleop_mask[0, 1] == 1.0
    env.set_teleop(0, 1, False)
    assert env.teleop_mask[0, 1] == 0.0
    assert env.teleop_vels[0, 1].abs().sum().item() == 0.0
