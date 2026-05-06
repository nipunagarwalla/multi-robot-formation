"""Device picking + a pure-torch radius_graph that works on CPU/MPS/CUDA.

torch_cluster's radius_graph is CPU-only (asserts x.is_cpu()), which would
otherwise force a host round-trip in the GNN forward. The replacement here
runs entirely in torch ops so the graph stays on whichever device the
trainer/inference is using.
"""
from __future__ import annotations

import torch


def pick_device(name: str = "auto") -> torch.device:
    """Resolve a device string. 'auto' picks cuda > mps > cpu.

    Accepted values: 'auto', 'cpu', 'cuda', 'cuda:0', 'mps'.
    """
    n = name.lower()
    if n == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if n.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {name} but torch.cuda.is_available() is False")
    if n == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("requested mps but torch.backends.mps.is_available() is False")
    return torch.device(n)


def radius_graph_torch(
    pos: torch.Tensor,
    r: float,
    batch: torch.Tensor,
    loop: bool = True,
    max_num_neighbors: int = 32,
) -> torch.Tensor:
    """Pure-torch drop-in for torch_cluster.radius_graph.

    pos:   (N, D) float positions
    batch: (N,)    int64 batch index — only nodes in the same batch can be
                   connected
    r:     scalar radius cutoff (Euclidean)
    loop:  include self-edges (i, i)

    Returns edge_index of shape (2, E) with rows [target, source], i.e. the
    same orientation torch_cluster.radius_graph emits when consumed by
    MessagePassing.

    The implementation is O(N^2) but n_agents is small here (4 per env, ~32
    in a typical batch) so cdist is the right tool. Honors max_num_neighbors
    by topk-truncating each row when needed (matches torch_cluster API).
    """
    n = pos.shape[0]
    if n == 0:
        return torch.empty(2, 0, dtype=torch.long, device=pos.device)
    d = torch.cdist(pos, pos)  # (N, N)
    same_batch = batch.unsqueeze(0) == batch.unsqueeze(1)
    mask = (d <= float(r)) & same_batch
    if not loop:
        eye = torch.eye(n, dtype=torch.bool, device=pos.device)
        mask = mask & ~eye

    # Cap per-target neighbors at max_num_neighbors using nearest-first.
    # We zero out everything beyond the top-k along each row of `mask`.
    if max_num_neighbors is not None and max_num_neighbors > 0 and n > max_num_neighbors:
        # Set non-eligible entries to +inf so they sort to the end
        d_for_sort = torch.where(mask, d, torch.full_like(d, float("inf")))
        # take indices of the closest max_num_neighbors per row
        _, topk_idx = torch.topk(d_for_sort, k=max_num_neighbors, dim=1, largest=False)
        keep = torch.zeros_like(mask)
        keep.scatter_(1, topk_idx, True)
        mask = mask & keep

    edges = mask.nonzero(as_tuple=False)  # (E, 2) with cols [target, source]
    return edges.t().contiguous()
