"""Exact-attention tests: dense vs SDPA reference, topk, and registry dispatch."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from ssa.attn import attn, available
from ssa.attn.dense import dense, topk
from ssa.attn.geometry import gqa_expand
from _fixtures import Geom, make_qkv


def _sdpa_reference(q, K, V):
    """Dense decode attention via PyTorch SDPA, GQA-expanded. Returns [H, d]."""
    H, d = q.shape
    K_exp = gqa_expand(K, H).permute(1, 0, 2)  # [H, n_k, d]
    V_exp = gqa_expand(V, H).permute(1, 0, 2)
    out = F.scaled_dot_product_attention(
        q.unsqueeze(1), K_exp, V_exp, is_causal=False  # query len 1
    )
    return out.squeeze(1)


def test_dense_matches_sdpa():
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=48), seed=0, dtype=torch.float64)
    out = dense(q, K, V)
    ref = _sdpa_reference(q, K, V)
    assert out.shape == (8, 16)
    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)


def test_topk_full_equals_dense():
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=32), seed=1, dtype=torch.float64)
    full = topk(q, K, V, k=32)
    torch.testing.assert_close(full, dense(q, K, V), rtol=1e-9, atol=1e-9)


def test_topk_matches_manual_renormalised_sum():
    q, K, V = make_qkv(Geom(H=3, H_kv=3, d=8, n_k=20), seed=2, dtype=torch.float64)
    from ssa.attn.geometry import attn_weights

    A = attn_weights(q, K)
    k = 5
    vals, idx = torch.topk(A, k, dim=-1)
    vals = vals / vals.sum(-1, keepdim=True)
    V_exp = gqa_expand(V, 3)
    expected = torch.zeros(3, 8, dtype=torch.float64)
    for h in range(3):
        for c in range(k):
            expected[h] += vals[h, c] * V_exp[idx[h, c], h]
    torch.testing.assert_close(topk(q, K, V, k=k), expected, rtol=1e-9, atol=1e-9)


def test_registry_dispatch_and_unknown():
    q, K, V = make_qkv(Geom(), seed=3, dtype=torch.float64)
    assert {"dense", "topk", "santa", "santa_strat", "santa_sys"} <= set(available())
    torch.testing.assert_close(attn(q, K, V, impl="dense"), dense(q, K, V))
    with pytest.raises(KeyError):
        attn(q, K, V, impl="nope")


def test_santa_estimators_shape_and_info():
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=64), seed=4, dtype=torch.float64)
    for impl in ("santa", "santa_strat", "santa_sys"):
        g = torch.Generator().manual_seed(0)
        out, info = attn(q, K, V, impl=impl, S=32, generator=g, return_info=True)
        assert out.shape == (4, 16)
        assert info.idx.shape == (4, 32)
        assert info.unique.shape == (4,)
        assert torch.all(info.unique <= 32) and torch.all(info.unique >= 1)
