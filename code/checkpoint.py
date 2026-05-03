"""Checkpoint save/load helpers.

The current format is a dict {"agent": ..., "optimizer": ..., "iteration": ...}
so a training run can resume with optimizer state intact. Older checkpoints
that were a bare state_dict still load — we treat them as iteration=0 with
no optimizer state.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(path: str, agent, optimizer, iteration: int) -> None:
    blob = {
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "iteration": int(iteration),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(blob, path)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    """Return {'agent': state_dict, 'optimizer': state_dict|None, 'iteration': int}.

    Accepts both the new dict format and the legacy bare state_dict format.
    """
    try:
        blob = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # weights_only=True can fail on older or pickled-object dicts.
        blob = torch.load(path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "agent" in blob and isinstance(blob["agent"], dict):
        return {
            "agent": blob["agent"],
            "optimizer": blob.get("optimizer"),
            "iteration": int(blob.get("iteration", 0)),
        }
    # Legacy: blob IS the agent state_dict
    return {"agent": blob, "optimizer": None, "iteration": 0}
