"""Smoke test for the sweep orchestration on a tiny CPU model (no weights/GPU)."""

from __future__ import annotations

import torch

from ssa.harness.ppl_sweep import _format_table, run_sweep
from _tiny_model import TinyCfg, tiny_model


def test_run_sweep_produces_all_conditions():
    cfg = TinyCfg()
    model = tiny_model(cfg, seed=1)
    g = torch.Generator().manual_seed(0)
    chunks = [torch.randint(0, cfg.vocab_size, (1, 24), generator=g) for _ in range(2)]

    conditions = [
        ("dense", {}),
        ("santa_sys", {"S": 8}),
        ("santa_hybrid", {"k_h": 4, "S": 12}),
    ]
    results = run_sweep(model, chunks, conditions=conditions, prefill_len=6, n_runs=2)

    assert len(results) == 3
    dense = results[0]
    assert dense["impl"] == "dense" and dense["read_fraction"] == 1.0 and dense["ppl_std"] == 0.0
    for r in results[1:]:
        assert 0.0 < r["read_fraction"] < 1.0
        assert r["ppl_mean"] > 0 and r["n_runs"] == 2
    assert results[2]["total_budget"] == 16  # k_h + S
    # table renders without error
    assert "condition" in _format_table(results)


def test_failing_condition_is_isolated_and_callback_fires():
    cfg = TinyCfg()
    model = tiny_model(cfg, seed=2)
    g = torch.Generator().manual_seed(0)
    chunks = [torch.randint(0, cfg.vocab_size, (1, 20), generator=g) for _ in range(1)]

    seen = []
    conditions = [("dense", {}), ("santa_bogus", {"S": 4}), ("santa_sys", {"S": 4})]
    results = run_sweep(
        model, chunks, conditions=conditions, prefill_len=6, n_runs=1,
        on_condition=lambda r, allr: seen.append(r["impl"]),
    )
    # the bad condition is recorded as an error; the sweep continues past it
    assert len(results) == 3
    assert "error" in results[1] and "santa_bogus" in results[1]["impl"]
    assert "error" not in results[2]  # later condition still ran
    assert seen == ["dense", "santa_bogus", "santa_sys"]  # callback fired each time
    assert "ERROR" in _format_table(results)
