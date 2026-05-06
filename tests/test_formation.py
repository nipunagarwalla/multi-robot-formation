"""Unit tests for the canonical formation slot positions and env smoke."""
import numpy as np
import pytest
import torch

from contract import FORMATION_SCALE, MAX_AGENTS, MAX_V, TELEOP_MAX_V
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


def test_teleop_action_uses_separate_speed_limit():
    env = FormationHallwayEnv({"num_envs": 1})
    env.vector_reset()
    env.set_teleop(0, 0, True)
    env.set_teleop_action(0, 0, np.array([TELEOP_MAX_V * 2.0, 0.0]))
    assert env.teleop_vels[0, 0, 0].item() == pytest.approx(TELEOP_MAX_V)
    assert env.teleop_vels[0, 0, 0].item() > MAX_V


def test_formation_reward_is_weighted_for_two_active_robots():
    env = FormationHallwayEnv({"num_envs": 1})
    env.vector_reset()
    env.ps[0] = torch.tensor(
        [[-0.30, 0.0], [0.30, 0.0], [0.0, 0.5], [0.0, -0.5]],
        dtype=torch.float32,
    )
    env.set_teleop(0, 2, True)
    env.set_teleop(0, 3, True)

    penalty, err = env._formation_reward(0, env.ps[0])
    # Active slots for scale=0.35 are at x=-0.175 and x=+0.175, so both
    # active robots are 12.5 cm from the assigned line slots.
    assert err == pytest.approx(0.125, abs=1e-5)
    expected = -2.0 * 2.25 * 0.125
    assert penalty[0].item() == pytest.approx(expected, abs=1e-5)
    assert penalty[1].item() == pytest.approx(expected, abs=1e-5)
    assert penalty[2].item() == 0.0
    assert penalty[3].item() == 0.0


def _copy_env_state(src: FormationHallwayEnv, dst: FormationHallwayEnv):
    dst.ps[:] = src.ps
    dst.goal_ps[:] = src.goal_ps
    dst.measured_vs[:] = src.measured_vs
    dst.teleop_mask[:] = src.teleop_mask
    dst.present_mask[:] = src.present_mask
    dst.teleop_vels[:] = src.teleop_vels
    dst.timesteps[:] = src.timesteps
    dst.goal_reached[:] = src.goal_reached
    dst.best_active_y[:] = src.best_active_y


def test_disturbance_rewards_do_not_change_clean_four_rewards():
    base = FormationHallwayEnv({"num_envs": 1})
    tuned = FormationHallwayEnv(
        {
            "num_envs": 1,
            "reward_coeffs": {
                "k_centroid_fwd_by_active": {4: 999.0},
                "k_center_by_active": {4: 999.0},
                "k_progress_best_by_active": {4: 999.0},
                "k_backward_by_active": {4: 999.0},
                "k_wall_proximity_by_active": {4: 999.0},
                "k_wall_contact_by_active": {4: 999.0},
                "k_teleop_chase": 999.0,
            },
        }
    )
    base.vector_reset()
    base.ps[0] = torch.tensor(
        [[-0.15, -1.0], [0.15, -1.0], [0.15, -0.7], [-0.15, -0.7]],
        dtype=torch.float32,
    )
    base.measured_vs.zero_()
    base.best_active_y[0] = base.ps[0, :, 1].mean()
    _copy_env_state(base, tuned)

    action = np.zeros((1, MAX_AGENTS, 2), dtype=np.float32)
    action[:, :, 1] = 1.0
    _, _, _, base_info = base.vector_step(action)
    _, _, _, tuned_info = tuned.vector_step(action)

    assert [base_info[0]["rewards"][i] for i in range(MAX_AGENTS)] == pytest.approx(
        [tuned_info[0]["rewards"][i] for i in range(MAX_AGENTS)]
    )
    assert tuned_info[0]["active_count"] == 4
    assert tuned_info[0]["backward_step"] is False


def test_disturbed_backward_progress_is_penalized():
    env = FormationHallwayEnv(
        {
            "num_envs": 1,
            "reward_coeffs": {
                "k_fwd": 0.0,
                "k_form": 0.0,
                "k_center_by_active": {3: 0.0},
                "k_centroid_fwd_by_active": {3: 0.0},
                "k_backward_by_active": {3: 10.0},
            },
        }
    )
    env.vector_reset()
    env.set_teleop(0, 3, True)
    action = np.zeros((1, MAX_AGENTS, 2), dtype=np.float32)
    action[0, :3, 1] = -1.0

    _, _, _, info = env.vector_step(action)

    assert info[0]["active_count"] == 3
    assert info[0]["backward_step"] is True
    assert sum(info[0]["rewards"].values()) < 0.0


def test_disturbed_best_progress_is_rewarded():
    env = FormationHallwayEnv(
        {
            "num_envs": 1,
            "reward_coeffs": {
                "k_fwd": 0.0,
                "k_form": 0.0,
                "k_center_by_active": {3: 0.0},
                "k_centroid_fwd_by_active": {3: 0.0},
                "k_progress_best_by_active": {3: 10.0},
            },
        }
    )
    env.vector_reset()
    env.set_teleop(0, 3, True)
    action = np.zeros((1, MAX_AGENTS, 2), dtype=np.float32)
    action[0, :3, 1] = 1.0

    _, _, _, info = env.vector_step(action)

    assert info[0]["active_count"] == 3
    assert sum(info[0]["rewards"].values()) > 0.0


def test_disturbed_wall_proximity_penalized_before_contact():
    env = FormationHallwayEnv(
        {
            "num_envs": 1,
            "reward_coeffs": {
                "k_fwd": 0.0,
                "k_form": 0.0,
                "k_center_by_active": {3: 0.0},
                "k_centroid_fwd_by_active": {3: 0.0},
                "k_wall_proximity_by_active": {3: 10.0},
                "k_wall_contact_by_active": {3: 0.0},
                "wall_safe_margin": 0.20,
            },
        }
    )
    env.vector_reset()
    env.set_teleop(0, 3, True)
    half_w = env.cfg["world_dim"][0] / 2.0 - env.cfg["agent_radius"]
    env.ps[0, :3, 0] = half_w - 0.05
    env.best_active_y[0] = env.ps[0, :3, 1].mean()

    _, _, _, info = env.vector_step(np.zeros((1, MAX_AGENTS, 2), dtype=np.float32))

    assert info[0]["wall_contact"] is False
    assert info[0]["min_wall_margin"] == pytest.approx(0.05, abs=1e-5)
    assert sum(info[0]["rewards"].values()) < 0.0


def test_disturbed_teleop_chase_is_penalized():
    env = FormationHallwayEnv(
        {
            "num_envs": 1,
            "reward_coeffs": {
                "k_fwd": 0.0,
                "k_form": 0.0,
                "k_center_by_active": {3: 0.0},
                "k_centroid_fwd_by_active": {3: 0.0},
                "k_teleop_chase": 20.0,
                "teleop_ignore_dist": 0.2,
                "teleop_ignore_lateral": 0.2,
            },
        }
    )
    env.vector_reset()
    env.ps[0] = torch.tensor(
        [[-0.3, 0.0], [0.0, 0.0], [0.3, 0.0], [0.7, -1.0]],
        dtype=torch.float32,
    )
    env.measured_vs.zero_()
    env.set_teleop(0, 3, True)
    action = np.zeros((1, MAX_AGENTS, 2), dtype=np.float32)
    action[0, :3] = np.array([0.5, -1.0], dtype=np.float32)

    _, _, _, info = env.vector_step(action)

    assert info[0]["active_count"] == 3
    assert info[0]["backward_step"] is True
    assert sum(info[0]["rewards"].values()) < 0.0
