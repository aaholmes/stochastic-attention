"""Contiguous-block sampling — `santa_block`, matched to how GPUs read memory.

A paged key-value (KV) cache reads coalesced contiguous blocks, not scattered rows. So instead
of sampling individual value rows, partition the keys into contiguous blocks of size
``B``, sample blocks ∝ their summed attention mass, and use each sampled block's
**exact within-block weighted average** of values:

    m_b  = Σ_{j∈b} A_j                      # block mass
    v̄_b  = (Σ_{j∈b} A_j V_j) / m_b          # exact within-block weighted mean
    out  = (1/S) Σ_s v̄_{b_s},  b_s ~ Categorical(m)
    E[out] = Σ_b m_b·v̄_b = Σ_j A_j V_j = AV   (unbiased for any partition)

Limits: ``B=1`` ⇒ iid ``santa`` (m=A, v̄=V); ``B≥n_k`` ⇒ ``dense`` (one block, v̄=AV,
zero variance). Block draws reuse the row samplers on the block-mass distribution,
so systematic/stratified compose for free.

Note: this is a *measurement* harness — it computes every block's v̄ then gathers the
sampled ones (simple + correct). A real kernel would fetch only sampled blocks; the
``BlockInfo`` read count reflects that (unique blocks × B).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..sampling.draws import (
    iid_indices,
    stratified_indices,
    systematic_indices,
    unique_counts,
)
from . import register
from .geometry import attn_weights, gqa_expand

_DRAW = {"sys": systematic_indices, "strat": stratified_indices, "iid": iid_indices}
_EPS = 1e-12


@dataclass
class BlockInfo:
    block_idx: torch.Tensor   # [H, S] sampled block indices
    unique_blocks: torch.Tensor  # [H] distinct blocks read
    B: int                    # block size
    reads: torch.Tensor       # [H] distinct value rows read = unique_blocks*B (≤ n_k)


def block_aggregate(A: torch.Tensor, V_exp: torch.Tensor, B: int):
    """Per-head block mass and exact within-block weighted value mean.

    A: ``[H, n_k]`` attention weights; V_exp: ``[n_k, H, d]``.
    Returns ``m`` ``[H, n_blocks]`` (sums to 1 per head) and ``vbar`` ``[H, n_blocks, d]``.
    """
    H, n_k = A.shape
    d = V_exp.shape[-1]
    n_blocks = (n_k + B - 1) // B
    pad = n_blocks * B - n_k

    A_pad = F.pad(A, (0, pad)) if pad else A                       # [H, n_blocks*B]
    m = A_pad.view(H, n_blocks, B).sum(-1)                         # [H, n_blocks]

    WV = A.unsqueeze(-1) * V_exp.permute(1, 0, 2)                  # [H, n_k, d]
    if pad:
        WV = F.pad(WV, (0, 0, 0, pad))
    W = WV.view(H, n_blocks, B, d).sum(2)                          # [H, n_blocks, d]
    vbar = W / m.clamp_min(_EPS).unsqueeze(-1)
    return m, vbar


@register("santa_block")
def santa_block(
    q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    S: int,
    B: int,
    method: str = "sys",
    generator: torch.Generator,
    return_info: bool = False,
    **cfg,
):
    """Sample ``S`` contiguous blocks ∝ block mass; average exact within-block means."""
    if method not in _DRAW:
        raise ValueError(f"unknown block sampler {method!r}; choose {list(_DRAW)}")
    H, d = q.shape
    n_k = K.shape[0]
    A = attn_weights(q, K)
    V_exp = gqa_expand(V, H)
    m, vbar = block_aggregate(A, V_exp, B)                        # [H,nb], [H,nb,d]

    blk = _DRAW[method](m, S, generator=generator)               # [H, S] block indices
    out = torch.gather(vbar, 1, blk.unsqueeze(-1).expand(H, S, d)).mean(dim=1)

    if return_info:
        uniq = unique_counts(blk)
        reads = (uniq * B).clamp_(max=n_k)
        return out, BlockInfo(block_idx=blk, unique_blocks=uniq, B=B, reads=reads)
    return out
