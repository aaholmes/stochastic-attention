"""Idea 1 — santa_hybrid: contract, known limits, and variance reduction.

The hybrid keeps the top-k_h keys exactly and samples the renormalized tail, so
it must (a) reduce to dense when the head covers everything, (b) reduce to the
pure sampler when the head is empty, and (c) cut variance vs plain sampling at a
matched read budget when attention mass is concentrated.
"""

from __future__ import annotations

import pytest
import torch

from ssa.attn import attn, available
from ssa.attn.dense import dense
from ssa.attn.hybrid import santa_hybrid
from ssa.harness.variance import collect_estimates, variance_trace
from _fixtures import Geom, make_qkv

TAILS = ["sys", "strat", "iid"]
_PURE = {"sys": "santa_sys", "strat": "santa_strat", "iid": "santa"}


# --- contract ----------------------------------------------------------------

def test_registered_and_dispatch():
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=64), seed=0, dtype=torch.float64)
    assert "santa_hybrid" in available()
    g = torch.Generator().manual_seed(0)
    out = attn(q, K, V, impl="santa_hybrid", k_h=4, S_tail=16, generator=g)
    assert out.shape == (4, 16)


def test_return_info():
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=64), seed=1, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    out, info = santa_hybrid(q, K, V, k_h=4, S_tail=16, generator=g, return_info=True)
    assert out.shape == (4, 16)
    assert info.idx.shape == (4, 16)
    assert info.unique.shape == (4,)


# --- known limits ------------------------------------------------------------

def test_full_head_equals_dense_and_is_deterministic():
    g = Geom(H=4, H_kv=2, d=16, n_k=32)
    q, K, V = make_qkv(g, seed=2, dtype=torch.float64)
    ref = dense(q, K, V)
    outs = []
    for seed in (0, 1, 2):
        gen = torch.Generator().manual_seed(seed)
        outs.append(santa_hybrid(q, K, V, k_h=g.n_k, S_tail=0, generator=gen))
    for o in outs:
        torch.testing.assert_close(o, ref, rtol=1e-9, atol=1e-9)
    # zero variance: identical across seeds
    torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=0)


@pytest.mark.parametrize("tail", TAILS)
def test_empty_head_equals_pure_sampler(tail):
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=64), seed=3, dtype=torch.float64)
    S = 24
    g_h = torch.Generator().manual_seed(42)
    g_p = torch.Generator().manual_seed(42)
    hyb = santa_hybrid(q, K, V, k_h=0, S_tail=S, tail=tail, generator=g_h)
    pure = attn(q, K, V, impl=_PURE[tail], S=S, generator=g_p)
    torch.testing.assert_close(hyb, pure, rtol=1e-9, atol=1e-9)


def test_peaked_head_no_nan():
    # Sharply peaked A (scaled scores): residual mass ~ 0 for a modest k_h.
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=64), seed=4, dtype=torch.float64)
    q = q * 25.0  # concentrate softmax mass
    g = torch.Generator().manual_seed(0)
    out = santa_hybrid(q, K, V, k_h=8, S_tail=8, generator=g)
    assert torch.isfinite(out).all()


# --- variance reduction (concentrated regime) --------------------------------

def _concentrated_qkv():
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=256), seed=7, dtype=torch.float64)
    return q * 6.0, K, V  # scale q to concentrate attention mass


def test_hybrid_beats_pure_sampling_at_matched_budget():
    q, K, V = _concentrated_qkv()
    S, k_h, runs = 32, 8, 600
    sys_var = variance_trace(
        collect_estimates("santa_sys", q, K, V, S=S, runs=runs, base_seed=0)
    )
    hyb_var = variance_trace(
        collect_estimates(
            "santa_hybrid", q, K, V, S=S - k_h, runs=runs, base_seed=0,
            k_h=k_h, tail="sys",
        )
    )
    assert hyb_var < sys_var, f"hybrid {hyb_var:.3e} not < santa_sys {sys_var:.3e}"


def test_reduction_grows_with_head_size():
    q, K, V = _concentrated_qkv()
    S, runs = 48, 600

    def hyb_var(k_h):
        return variance_trace(
            collect_estimates(
                "santa_hybrid", q, K, V, S=S - k_h, runs=runs, base_seed=0,
                k_h=k_h, tail="sys",
            )
        )

    assert hyb_var(16) <= hyb_var(4) * 1.05, "more head should not increase variance"
