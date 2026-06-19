"""Adapter + installer tests: the decode seam faithfully runs ssa estimators.

Runs on a tiny CPU model (no weights, no GPU) so it gates in CI.
"""

from __future__ import annotations

import torch

from ssa.attn import attn as ssa_attn
from ssa.models.patch import ReadStats, install, make_decode_op, uninstall
from _tiny_model import TinyCfg, tiny_model


# --- adapter geometry (no model) ---------------------------------------------

def test_adapter_matches_direct_dense():
    H, H_kv, d, n_k = 4, 2, 8, 12
    torch.manual_seed(0)
    q = torch.randn(1, H, 1, d)
    full_k = torch.randn(1, H_kv, n_k, d)
    full_v = torch.randn(1, H_kv, n_k, d)

    op = make_decode_op("dense", base_seed=0, stats=ReadStats(), cfg={})
    got = op(q, full_k, full_v, scale=d ** -0.5, layer_idx=0)

    qd = q[0, :, 0, :]
    K = full_k[0].permute(1, 0, 2)
    V = full_v[0].permute(1, 0, 2)
    want = ssa_attn(qd, K, V, impl="dense").unsqueeze(0).unsqueeze(2)
    assert got.shape == (1, H, 1, d)
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


def test_read_stats_dense_reads_everything():
    H, H_kv, d, n_k = 4, 2, 8, 10
    q = torch.randn(1, H, 1, d)
    fk = torch.randn(1, H_kv, n_k, d)
    fv = torch.randn(1, H_kv, n_k, d)
    stats = ReadStats()
    op = make_decode_op("dense", base_seed=0, stats=stats, cfg={})
    op(q, fk, fv, scale=d ** -0.5, layer_idx=0)
    assert stats.read_fraction == 1.0
    assert stats.avg_n_k == n_k


def test_read_stats_topk_reads_k():
    H, H_kv, d, n_k = 4, 2, 8, 40
    q = torch.randn(1, H, 1, d)
    fk = torch.randn(1, H_kv, n_k, d)
    fv = torch.randn(1, H_kv, n_k, d)
    stats = ReadStats()
    op = make_decode_op("topk", base_seed=0, stats=stats, cfg={"k": 4})
    op(q, fk, fv, scale=d ** -0.5, layer_idx=0)
    assert stats.avg_reads == 4  # exactly the top-k rows, not all n_k
    assert stats.read_fraction == 4 / n_k


def test_read_stats_sampler_reads_fewer():
    H, H_kv, d, n_k = 4, 2, 8, 64
    q = torch.randn(1, H, 1, d)
    fk = torch.randn(1, H_kv, n_k, d)
    fv = torch.randn(1, H_kv, n_k, d)
    stats = ReadStats()
    op = make_decode_op("santa_sys", base_seed=1, stats=stats, cfg={"S": 8})
    op(q, fk, fv, scale=d ** -0.5, layer_idx=0)
    assert 0.0 < stats.read_fraction < 1.0
    assert stats.avg_reads <= 8  # unique keys ≤ S


def test_read_fraction_never_exceeds_one():
    # Degenerate: k_h larger than n_k must still report read_fraction ≤ 1.
    H, H_kv, d, n_k = 4, 2, 8, 10
    q = torch.randn(1, H, 1, d)
    fk = torch.randn(1, H_kv, n_k, d)
    fv = torch.randn(1, H_kv, n_k, d)
    stats = ReadStats()
    op = make_decode_op("santa_hybrid", base_seed=0, stats=stats, cfg={"k_h": 32, "S": 16})
    op(q, fk, fv, scale=d ** -0.5, layer_idx=0)
    assert stats.read_fraction <= 1.0


# --- engine equivalence (tiny model) -----------------------------------------

def _decode_logits(model, ids, P):
    """Prefill ids[:,:P], then decode the rest one token at a time; stack logits."""
    cache = model.alloc_cache(ids.shape[1] + 1)
    with torch.inference_mode():
        model(ids[:, :P], cache, start_pos=0)
        outs = []
        for t in range(P, ids.shape[1]):
            logits = model(ids[:, t:t + 1], cache)
            outs.append(logits[:, -1, :].clone())
    return torch.stack(outs, dim=1)


def test_install_dense_matches_hook_off():
    cfg = TinyCfg()
    model = tiny_model(cfg, seed=3)
    torch.manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    P = 5

    ref = _decode_logits(model, ids, P)

    stats = install(model, "dense")
    hooked = _decode_logits(model, ids, P)
    uninstall(model)

    torch.testing.assert_close(hooked, ref, rtol=1e-4, atol=1e-4)
    assert stats.read_fraction == 1.0
    # uninstall restored the default path
    after = _decode_logits(model, ids, P)
    torch.testing.assert_close(after, ref, rtol=1e-4, atol=1e-4)


def test_install_sampler_runs_and_reads_fewer():
    cfg = TinyCfg()
    model = tiny_model(cfg, seed=4)
    torch.manual_seed(8)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))

    stats = install(model, "santa_sys", base_seed=0, S=4)
    logits = _decode_logits(model, ids, P=6)
    uninstall(model)

    assert torch.isfinite(logits).all()
    assert 0.0 < stats.read_fraction < 1.0
    # one op call per (layer, decode-step): 2 layers * 10 steps
    assert stats.steps == cfg.num_hidden_layers * (16 - 6)
