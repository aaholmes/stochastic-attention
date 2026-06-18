"""Index draws for the three sampling estimators (design doc §2).

All operate per head on attention weights ``A`` of shape ``[H, n_k]`` and return
sampled key indices of shape ``[H, S]``. Sampling is **with replacement**; use
``unique_counts`` to recover the per-head unique-key count (paper App. M).

  - ``iid``: S indices ~ Categorical(A).
  - ``stratified``: S equal-mass CDF strata, one independent offset per stratum.
  - ``systematic``: a single shared offset U ~ Unif[0, 1/S) per head, thresholds
    ``U + m/S`` — exactly one RNG draw per head.
"""

from __future__ import annotations

import torch

from .cdf import build_cdf, invert_cdf


def iid_indices(A: torch.Tensor, S: int, *, generator: torch.Generator) -> torch.Tensor:
    """``[H, S]`` indices drawn i.i.d. from Categorical(A)."""
    return torch.multinomial(A, S, replacement=True, generator=generator)


def stratified_indices(A: torch.Tensor, S: int, *, generator: torch.Generator) -> torch.Tensor:
    """``[H, S]`` indices, one per equal-mass stratum, independent offsets.

    Threshold for stratum ``m``: ``T_m ~ Unif[m/S, (m+1)/S)`` drawn independently
    per head and stratum (H*S random numbers total).
    """
    H = A.shape[0]
    F = build_cdf(A)
    m = torch.arange(S, device=A.device, dtype=A.dtype)
    noise = torch.rand(H, S, generator=generator, device=A.device, dtype=A.dtype)
    T = (m + noise) / S
    return invert_cdf(F, T)


def systematic_indices(A: torch.Tensor, S: int, *, generator: torch.Generator) -> torch.Tensor:
    """``[H, S]`` indices from a single shared offset per head.

    One draw ``U ~ Unif[0, 1/S)`` per head (H random numbers total), thresholds
    ``T_m = U + m/S``.
    """
    H = A.shape[0]
    F = build_cdf(A)
    m = torch.arange(S, device=A.device, dtype=A.dtype)
    U = torch.rand(H, 1, generator=generator, device=A.device, dtype=A.dtype) / S
    T = U + m / S
    return invert_cdf(F, T)


def unique_counts(idx: torch.Tensor) -> torch.Tensor:
    """Per-head count of distinct sampled indices. ``idx``: ``[H, S]`` -> ``[H]``."""
    H = idx.shape[0]
    return torch.tensor(
        [int(idx[h].unique().numel()) for h in range(H)],
        device=idx.device,
    )
