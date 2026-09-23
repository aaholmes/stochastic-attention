"""Tests for the variance-convergence curve builder + plot writer."""

from __future__ import annotations

import torch

from ssa.harness.plot_variance import build_curves, plot_curves
from _fixtures import Geom, make_qkv


def test_build_curves_shapes_and_hybrid_budget():
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=128), seed=0, dtype=torch.float64)
    q = q * 4.0  # concentrate mass so hybrid has a head to exploit
    budgets = [8, 16, 32, 64]
    curves = build_curves(q, K, V, budgets=budgets, hybrid_k_h=(4,), runs=300)

    # plain samplers span the full budget list; hybrid starts above its k_h
    assert curves["santa_sys"]["x"] == budgets
    assert curves["santa_hybrid(k_h=4)"]["x"] == [8, 16, 32, 64]  # all > 4
    for c in curves.values():
        assert len(c["x"]) == len(c["var"])
        assert all(v > 0 for v in c["var"])
        assert c["slope"] < 0  # variance decreases with budget


def test_hybrid_x_excludes_budgets_below_head():
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=128), seed=1, dtype=torch.float64)
    curves = build_curves(q, K, V, budgets=[8, 16, 32], hybrid_k_h=(16,), runs=200)
    # k_h=16 leaves no tail at budget 8 or 16, so only 32 (total) survives
    assert curves["santa_hybrid(k_h=16)"]["x"] == [32]


def test_plot_writes_png(tmp_path):
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=128), seed=2, dtype=torch.float64)
    curves = build_curves(q, K, V, budgets=[8, 16, 32], hybrid_k_h=(4,), runs=150)
    out = tmp_path / "conv.png"
    plot_curves(curves, out)
    assert out.exists() and out.stat().st_size > 0
