"""Tests for skip_k (magnitude-ranked cluster selection)."""

from __future__ import annotations

import torch

from ssa.attn import attn, available
from ssa.attn.dense import dense
from ssa.attn.skip_k import skip_k
from _fixtures import Geom, make_qkv


def test_registered():
    assert "skip_k" in available()


def test_full_coverage_equals_dense():
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=64), seed=0, dtype=torch.float64)
    out = skip_k(q, K, V, coverage=1.0, B=16)
    torch.testing.assert_close(out, dense(q, K, V), rtol=1e-6, atol=1e-6)


def test_low_coverage_reads_fewer_and_finite():
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=128), seed=1, dtype=torch.float64)
    q = q * 4.0  # concentrate attention so few clusters cover the mass
    out, info = skip_k(q, K, V, coverage=0.9, B=16, return_info=True)
    assert out.shape == (8, 16)
    assert torch.isfinite(out).all()
    assert info.unique.shape == (8,)
    assert torch.all(info.unique < 128) and torch.all(info.unique >= 1)


def test_coverage_monotone_in_reads():
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=128), seed=2, dtype=torch.float64)
    q = q * 4.0
    _, lo = skip_k(q, K, V, coverage=0.8, B=16, return_info=True)
    _, hi = skip_k(q, K, V, coverage=0.999, B=16, return_info=True)
    assert lo.unique.float().mean() <= hi.unique.float().mean()


def test_dispatch_and_generator_ignored():
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=48), seed=3, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    # generator is accepted (sampling-path signature) but ignored — deterministic
    a = attn(q, K, V, impl="skip_k", coverage=0.9, generator=g)
    b = attn(q, K, V, impl="skip_k", coverage=0.9)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
