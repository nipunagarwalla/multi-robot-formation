"""Tests for training-time best-checkpoint eval helpers."""

from train_hallway import _score_eval


def test_score_eval_prioritizes_worst_regime_success():
    stable = {
        4: {"success_rate": 0.9, "mean_forward_velocity": 0.6, "mean_formation_error": 0.03, "mean_collisions": 0.0},
        3: {"success_rate": 0.9, "mean_forward_velocity": 0.5, "mean_formation_error": 0.04, "mean_collisions": 0.0},
        2: {"success_rate": 0.9, "mean_forward_velocity": 0.4, "mean_formation_error": 0.05, "mean_collisions": 0.0},
    }
    uneven = {
        4: {"success_rate": 1.0, "mean_forward_velocity": 0.8, "mean_formation_error": 0.01, "mean_collisions": 0.0},
        3: {"success_rate": 1.0, "mean_forward_velocity": 0.8, "mean_formation_error": 0.01, "mean_collisions": 0.0},
        2: {"success_rate": 0.5, "mean_forward_velocity": 0.8, "mean_formation_error": 0.01, "mean_collisions": 0.0},
    }

    assert _score_eval(stable) > _score_eval(uneven)


def test_score_eval_penalizes_wall_contact_and_backward_steps():
    clean = {
        4: {"success_rate": 1.0, "mean_forward_velocity": 0.6, "mean_formation_error": 0.03, "mean_collisions": 0.0},
        3: {"success_rate": 1.0, "mean_forward_velocity": 0.5, "mean_formation_error": 0.04, "mean_collisions": 0.0},
        2: {"success_rate": 1.0, "mean_forward_velocity": 0.4, "mean_formation_error": 0.05, "mean_collisions": 0.0},
    }
    wall_dragging = {
        k: {
            **v,
            "mean_wall_hits": 12.0,
            "mean_wall_contact_steps": 40.0,
            "mean_backward_steps": 8.0,
            "mean_min_wall_margin": 0.01,
        }
        for k, v in clean.items()
    }

    assert _score_eval(clean) > _score_eval(wall_dragging)
