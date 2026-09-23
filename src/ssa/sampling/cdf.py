"""Per-head cumulative distribution function (CDF) construction and inverse-CDF lookup.

The sampling estimators all reduce to: build the cumulative distribution ``F``
from the attention weights ``A``, then map threshold values in ``[0, 1)`` back to
key indices via ``searchsorted`` (the inverse CDF ``F⁻¹``).
"""

from __future__ import annotations

import torch


def build_cdf(A: torch.Tensor) -> torch.Tensor:
    """Cumulative distribution per head.

    A: ``[H, n_k]`` non-negative rows summing to 1 (a softmax output).
    Returns ``F`` of the same shape, monotone non-decreasing along the last dim
    with ``F[..., -1] == 1`` (the last entry is pinned to 1 to absorb roundoff).
    """
    F = torch.cumsum(A, dim=-1)
    F[..., -1] = 1.0
    return F


def invert_cdf(F: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Inverse CDF: smallest index ``j`` per row with ``F[j] >= T``.

    F: ``[H, n_k]`` cumulative distribution (rows non-decreasing).
    T: ``[H, S]`` threshold values in ``[0, 1)``.
    Returns ``[H, S]`` long indices in ``[0, n_k)``.
    """
    n_k = F.shape[-1]
    idx = torch.searchsorted(F, T, right=False)
    return idx.clamp_(max=n_k - 1)
