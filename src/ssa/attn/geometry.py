"""Decode-time attention geometry + weight computation (design doc §2).

Single query per head ``q=[H, d]`` against a GQA KV cache ``K,V=[n_k, H_kv, d]``
with group size ``G = H // H_kv``. At decode the single query is causally allowed
to see every cached position, so there is no mask to apply.

Softmax is accumulated in at least float32 (matching the engine's RMSNorm/softmax
idiom in ``../llms/src/engine/attention.py``) so low-precision inputs don't bias
the weights.
"""

from __future__ import annotations

import math

import torch


def _accum_dtype(dtype: torch.dtype) -> torch.dtype:
    """Upcast low-precision dtypes to float32 for the softmax; keep float64."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def gqa_expand(T: torch.Tensor, H: int) -> torch.Tensor:
    """Expand a KV tensor ``[n_k, H_kv, d]`` to per-query-head ``[n_k, H, d]``.

    Query head ``h`` reads KV head ``h // G`` with ``G = H // H_kv`` — the
    repeat-interleave grouping used by Llama/Qwen GQA.
    """
    n_k, H_kv, d = T.shape
    G = H // H_kv
    kv_index = torch.arange(H, device=T.device) // G
    return T[:, kv_index, :]


def attn_weights(q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Per-head attention weights ``A = softmax(qKᵀ/√d)`` of shape ``[H, n_k]``."""
    H, d = q.shape
    K_exp = gqa_expand(K, H)  # [n_k, H, d]
    acc = _accum_dtype(q.dtype)
    scores = torch.einsum("hd,jhd->hj", q.to(acc), K_exp.to(acc)) / math.sqrt(d)
    return torch.softmax(scores, dim=-1)


def weighted_sum(weights: torch.Tensor, V_exp: torch.Tensor) -> torch.Tensor:
    """``out[h] = Σ_j weights[h,j] V_exp[j,h]`` -> ``[H, d]``."""
    return torch.einsum("hj,jhd->hd", weights.to(V_exp.dtype), V_exp)


def gather_rows(V_exp: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather sampled value rows. ``V_exp=[n_k,H,d]``, ``idx=[H,S]`` -> ``[H,S,d]``."""
    H, S = idx.shape
    d = V_exp.shape[-1]
    V_perm = V_exp.permute(1, 0, 2)  # [H, n_k, d]
    return torch.gather(V_perm, 1, idx.unsqueeze(-1).expand(H, S, d))
