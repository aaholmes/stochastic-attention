"""Tests for the sys-vs-hybrid concentration crossover probe."""

from __future__ import annotations

from ssa.harness.crossover import _hybrid_advantage, _measure, participation_ratio, _qkv


def test_participation_ratio_tracks_scale():
    # Higher q-scale → peakier softmax → fewer effective tokens.
    q_flat, K_flat, _ = _qkv(0.2, n_k=256, H=4, d=16, seed=0)
    q_peak, K_peak, _ = _qkv(4.0, n_k=256, H=4, d=16, seed=0)
    assert participation_ratio(q_flat, K_flat) > participation_ratio(q_peak, K_peak)


def test_measure_returns_read_fraction_and_variance():
    q, K, V = _qkv(2.0, n_k=256, H=4, d=16, seed=1)
    rf, var = _measure("santa_sys", q, K, V, runs=100, base_seed=0, S=16)
    assert 0 < rf <= 1 and var > 0
    # more samples → lower variance, more reads
    rf2, var2 = _measure("santa_sys", q, K, V, runs=100, base_seed=0, S=64)
    assert rf2 > rf and var2 < var


def test_hybrid_advantage_sign():
    # A constructed row where hybrid is uniformly lower variance → advantage < 1.
    row = {"sys": [(0.01, 10.0), (0.02, 5.0)], "hybrid_kh1": [(0.01, 4.0), (0.02, 2.0)]}
    assert _hybrid_advantage(row) < 1.0
    row2 = {"sys": [(0.01, 1.0), (0.02, 0.5)], "hybrid_kh1": [(0.01, 4.0), (0.02, 2.0)]}
    assert _hybrid_advantage(row2) > 1.0
