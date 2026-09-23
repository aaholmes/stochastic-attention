"""Semi-stochastic estimators: ``santa`` (iid), ``santa_strat``, ``santa_sys``.

Each draws ``S`` value rows (with replacement) from the per-head attention
distribution and returns their simple average ``(1/S) Σ V[J_s]`` — an unbiased
estimate of ``dense``. The three differ only in how
the ``S`` indices are drawn (see ``ssa.sampling.draws``).

Pass ``return_info=True`` to also get the sampled indices and per-head unique-key
count (useful for the with-replacement caching analysis, paper App. M).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..sampling.draws import (
    iid_indices,
    stratified_indices,
    systematic_indices,
    unique_counts,
)
from . import register
from .geometry import attn_weights, gather_rows, gqa_expand


@dataclass
class SantaInfo:
    """Side-channel from a sampled estimate."""

    idx: torch.Tensor       # [H, S] sampled key indices
    unique: torch.Tensor    # [H] distinct keys actually read


def _estimate(q, K, V, S, draw_fn, generator, return_info):
    H = q.shape[0]
    A = attn_weights(q, K)
    idx = draw_fn(A, S, generator=generator)        # [H, S]
    V_exp = gqa_expand(V, H)
    sampled = gather_rows(V_exp, idx)               # [H, S, d]
    out = sampled.mean(dim=1)                       # [H, d]
    if return_info:
        return out, SantaInfo(idx=idx, unique=unique_counts(idx))
    return out


@register("santa")
def santa(q, K, V, *, S: int, generator: torch.Generator, return_info: bool = False, **cfg):
    """i.i.d. Categorical(A) sampling, averaged."""
    return _estimate(q, K, V, S, iid_indices, generator, return_info)


@register("santa_strat")
def santa_strat(q, K, V, *, S: int, generator: torch.Generator, return_info: bool = False, **cfg):
    """Stratified sampling: one independent draw per equal-mass stratum."""
    return _estimate(q, K, V, S, stratified_indices, generator, return_info)


@register("santa_sys")
def santa_sys(q, K, V, *, S: int, generator: torch.Generator, return_info: bool = False, **cfg):
    """Systematic sampling: one shared offset per head."""
    return _estimate(q, K, V, S, systematic_indices, generator, return_info)
