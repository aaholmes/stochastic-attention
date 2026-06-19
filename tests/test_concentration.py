"""Tests for the attention-concentration diagnostic."""

from __future__ import annotations

import math

import torch

from ssa.harness.concentration import ConcentrationStats, measure_concentration
from _tiny_model import TinyCfg, tiny_model


def test_stats_uniform_vs_peaked():
    n_k = 64
    uniform = torch.full((2, n_k), 1.0 / n_k)
    peaked = torch.zeros(2, n_k)
    peaked[:, 0] = 0.97
    peaked[:, 1:] = 0.03 / (n_k - 1)

    su, sp = ConcentrationStats(), ConcentrationStats()
    su.record(uniform)
    sp.record(peaked)
    u, p = su.summary(), sp.summary()

    # uniform: participation ratio ≈ n_k, max entropy; peaked: PR ≈ 1, low entropy
    assert abs(u["mean_participation_ratio"] - n_k) < 1.0
    assert p["mean_participation_ratio"] < 2.0
    assert u["mean_entropy_nats"] > p["mean_entropy_nats"]
    assert math.isclose(u["mean_entropy_nats"], math.log(n_k), rel_tol=1e-3)

    # collisions: peaked distribution reads far fewer unique rows at the same budget
    assert p["expected_read_fraction"][32] < u["expected_read_fraction"][32]
    # top-1 mass reflects concentration
    assert p["mean_top_mass"][1] > 0.9 and u["mean_top_mass"][1] < 0.1


def test_expected_read_fraction_monotone_in_budget():
    A = torch.softmax(torch.randn(4, 128), dim=-1)
    s = ConcentrationStats()
    s.record(A)
    rf = s.summary()["expected_read_fraction"]
    assert rf[8] < rf[32] < rf[128]          # more samples → more unique reads
    assert all(0 < v <= 1 for v in rf.values())


def test_measure_concentration_on_tiny_model():
    cfg = TinyCfg()
    model = tiny_model(cfg, seed=1)
    g = torch.Generator().manual_seed(0)
    chunks = [torch.randint(0, cfg.vocab_size, (1, 16), generator=g) for _ in range(2)]

    summary = measure_concentration(model, chunks, prefill_len=6)
    assert summary["records"] > 0
    pr = summary["mean_participation_ratio"]
    assert 1.0 <= pr <= cfg.num_attention_heads * cfg.head_dim + cfg.max_position_embeddings
    # diagnostic must restore the model (no op left installed)
    from engine.attention import Attention
    assert all(m.decode_attn_op is None for m in model.modules() if isinstance(m, Attention))
