"""Tests for model-side teleop masking helpers."""
import torch

from model import _filter_edges_by_source


def test_filter_edges_by_source_removes_teleop_sources():
    # Edge orientation follows radius_graph_torch/AFOR convention:
    # row 0 is target, row 1 is source.
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 1, 0, 1, 2, 3, 2, 3],
        ],
        dtype=torch.long,
    )
    source_node_mask = torch.tensor([[True, False, True, False]])

    filtered = _filter_edges_by_source(edge_index, source_node_mask)

    assert filtered.tolist() == [
        [0, 1, 2, 3],
        [0, 0, 2, 2],
    ]
