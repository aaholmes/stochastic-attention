"""Sanity tests for the variance-vs-block-size curve builder."""

from __future__ import annotations

import torch

from ssa.attn import attn
from ssa.harness.plot_blocks import build_block_curve, plot_block_curve
from ssa.harness.variance import collect_estimates, variance_trace
from _fixtures import Geom, make_qkv


def test_curve_shapes_and_finite():
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=128), seed=0, dtype=torch.float64)
    q = q * 4.0
    rows = build_block_curve(q, K, V, read_budget=64, B_values=[1, 4, 16, 64], runs=200)
    assert [r["B"] for r in rows] == [1, 4, 16, 64]
    for r in rows:
        assert r["variance"] > 0 and 0 < r["read_fraction"] <= 1
        assert r["S"] == max(1, 64 // r["B"])


def test_large_block_lowers_variance_toward_exact():
    # At fixed budget, coarser blocks fold more mass into exact within-block means.
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=128), seed=1, dtype=torch.float64)
    q = q * 4.0
    rows = build_block_curve(q, K, V, read_budget=64, B_values=[1, 128], runs=300)
    var = {r["B"]: r["variance"] for r in rows}
    assert var[128] < var[1]  # B=n_k is exact (≈0) vs row-level sampling


def test_b1_matches_row_level_santa():
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=128), seed=2, dtype=torch.float64)
    rows = build_block_curve(q, K, V, read_budget=32, B_values=[1], runs=400, base_seed=0)
    # santa_block(B=1, sys) ≡ santa_sys at the same S/seed → same variance
    sys_var = variance_trace(collect_estimates("santa_sys", q, K, V, S=32, runs=400, base_seed=0))
    assert abs(rows[0]["variance"] - sys_var) / sys_var < 1e-9


def test_plot_writes_png(tmp_path):
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=128), seed=3, dtype=torch.float64)
    rows = build_block_curve(q, K, V, read_budget=32, B_values=[1, 8, 64], runs=150)
    out = tmp_path / "vb.png"
    plot_block_curve(rows, out, title="t")
    assert out.exists() and out.stat().st_size > 0
