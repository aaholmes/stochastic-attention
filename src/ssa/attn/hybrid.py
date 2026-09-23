"""Hybrid estimator ``santa_hybrid``: deterministic head + sampled tail.

Keep the top-``k_h`` keys per head **exactly** (zero variance), then form an
unbiased Monte-Carlo estimate of the renormalized residual ("tail"):

    head_out = Σ_{j∈top}  A[j] V[j]                 # exact
    out      = head_out + (1 − m_head) · tail_avg   # tail_avg ~ residual dist.

E[out] = AV for any distribution. Variance comes only from the tail, so it shrinks
as the head absorbs more mass — the gain grows with ``m_head``.

The tail sample budget may be given as ``S_tail`` (direct) or ``S`` (the generic
name used by ``ssa.harness.variance.collect_estimates``); ``S_tail`` wins if both.
"""

from __future__ import annotations

import torch

from ..sampling.draws import (
    iid_indices,
    stratified_indices,
    systematic_indices,
    unique_counts,
)
from . import register
from .geometry import attn_weights, gather_rows, gqa_expand
from .santa import SantaInfo

_TAIL_DRAW = {
    "sys": systematic_indices,
    "strat": stratified_indices,
    "iid": iid_indices,
}
_EPS = 1e-12


@register("santa_hybrid")
def santa_hybrid(
    q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    k_h: int,
    S_tail: int | None = None,
    S: int | None = None,
    tail: str = "sys",
    generator: torch.Generator,
    return_info: bool = False,
    **cfg,
):
    """Top-k_h exact head + sampled renormalized tail. Unbiased for any A."""
    n_tail = S_tail if S_tail is not None else S
    if n_tail is None:
        raise TypeError("santa_hybrid needs a tail budget via S_tail= or S=")
    if tail not in _TAIL_DRAW:
        raise ValueError(f"unknown tail sampler {tail!r}; choose {list(_TAIL_DRAW)}")

    H, d = q.shape
    A = attn_weights(q, K)              # [H, n_k]
    n_k = A.shape[-1]
    k_h = min(max(k_h, 0), n_k)
    V_exp = gqa_expand(V, H)            # [n_k, H, d]
    V_perm = V_exp.permute(1, 0, 2)     # [H, n_k, d]

    # --- exact head ----------------------------------------------------------
    if k_h > 0:
        top_vals, top_idx = torch.topk(A, k_h, dim=-1)          # [H, k_h]
        V_top = torch.gather(V_perm, 1, top_idx.unsqueeze(-1).expand(H, k_h, d))
        head_out = torch.einsum("hk,hkd->hd", top_vals.to(V_top.dtype), V_top)
        A_tail = A.scatter(1, top_idx, 0.0)                     # head mass removed
    else:
        head_out = torch.zeros(H, d, dtype=V_exp.dtype, device=V_exp.device)
        A_tail = A

    residual = A_tail.sum(dim=-1)                               # [H] = 1 − m_head
    out = head_out

    # --- sampled tail --------------------------------------------------------
    info_idx = torch.empty(H, 0, dtype=torch.long, device=A.device)
    if n_tail > 0:
        safe = residual > _EPS                                 # [H]
        # rows whose head holds ~all mass: give them a valid (uniform) tail dist
        # so the sampler never sees an all-zero row, then mask their contribution.
        uniform = torch.full_like(A_tail, 1.0 / n_k)
        A_dist = torch.where(safe.unsqueeze(-1), A_tail, uniform)
        A_dist = A_dist / A_dist.sum(dim=-1, keepdim=True)
        idx = _TAIL_DRAW[tail](A_dist, n_tail, generator=generator)   # [H, n_tail]
        tail_avg = gather_rows(V_exp, idx).mean(dim=1)               # [H, d]
        tail_term = residual.unsqueeze(-1).to(tail_avg.dtype) * tail_avg
        tail_term = torch.where(safe.unsqueeze(-1), tail_term, torch.zeros_like(tail_term))
        out = out + tail_term
        info_idx = idx

    if return_info:
        return out, SantaInfo(idx=info_idx, unique=unique_counts(info_idx))
    return out
