"""Decode-PPL harness tests on a tiny CPU model (no weights, no GPU).

Validates the loop itself (dense via the seam == hook-off) and that sampled
attention gives a finite PPL that converges to the dense baseline as the sample
budget grows.
"""

from __future__ import annotations

import torch

from ssa.harness.perplexity import decode_ppl
from ssa.models.patch import install, uninstall
from _tiny_model import TinyCfg, tiny_model


def _chunks(cfg, n_chunks=3, length=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, cfg.vocab_size, (1, length), generator=g) for _ in range(n_chunks)]


def test_dense_seam_matches_hook_off():
    cfg = TinyCfg()
    model = tiny_model(cfg, seed=1)
    chunks = _chunks(cfg)

    ref = decode_ppl(model, chunks, prefill_len=6)

    install(model, "dense")
    hooked = decode_ppl(model, chunks, prefill_len=6)
    uninstall(model)

    assert abs(hooked["ppl"] - ref["ppl"]) / ref["ppl"] < 1e-4
    assert hooked["token_count"] == ref["token_count"]


def test_sampler_ppl_finite_and_converges_to_dense():
    cfg = TinyCfg()
    model = tiny_model(cfg, seed=2)
    chunks = _chunks(cfg, length=24, seed=5)

    dense = decode_ppl(model, chunks, prefill_len=6)["ppl"]

    install(model, "santa_sys", base_seed=0, S=4)
    small = decode_ppl(model, chunks, prefill_len=6)["ppl"]
    uninstall(model)

    install(model, "santa_sys", base_seed=0, S=256)
    large = decode_ppl(model, chunks, prefill_len=6)["ppl"]
    uninstall(model)

    assert small > 0 and large > 0
    # large budget drives the estimate to the true AV, so PPL approaches dense
    assert abs(large - dense) / dense < 0.05
    # and a large budget is at least as close to dense as a tiny one
    assert abs(large - dense) <= abs(small - dense) + 1e-6


def test_read_fraction_recorded():
    cfg = TinyCfg()
    model = tiny_model(cfg, seed=3)
    chunks = _chunks(cfg, length=24)

    stats = install(model, "santa_hybrid", base_seed=0, k_h=2, S=4)
    decode_ppl(model, chunks, prefill_len=8)
    uninstall(model)

    assert 0.0 < stats.read_fraction < 1.0
    assert stats.steps > 0
