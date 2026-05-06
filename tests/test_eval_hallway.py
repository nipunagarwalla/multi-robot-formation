"""Tests for eval helpers."""
import numpy as np

from env_hallway import FormationHallwayEnv
from eval_hallway import apply_fixed_active_count


def test_apply_fixed_active_count_holds_inactive_robots():
    env = FormationHallwayEnv({"num_envs": 1})
    env.vector_reset()

    apply_fixed_active_count(env, 0, active_count=2)

    np.testing.assert_allclose(env.teleop_mask[0].cpu().numpy(), [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(env.teleop_vels[0, 2:].cpu().numpy(), 0.0)
