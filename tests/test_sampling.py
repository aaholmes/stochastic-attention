"""Sampling-primitive unit tests (CDF + inverse + index draws).

These are the load-bearing mechanics of the paper (design doc §2), so they are
tested in isolation on hand-checkable distributions before any estimator uses them.
"""

from __future__ import annotations

import pytest
import torch

from ssa.sampling.cdf import build_cdf, invert_cdf
from ssa.sampling.draws import (
    iid_indices,
    stratified_indices,
    systematic_indices,
    unique_counts,
)


def _softmax_weights(H: int, n_k: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    scores = torch.randn(H, n_k, generator=gen, dtype=torch.float64)
    return torch.softmax(scores, dim=-1)


def _uniform_weights(H: int, n_k: int) -> torch.Tensor:
    return torch.full((H, n_k), 1.0 / n_k, dtype=torch.float64)


# --- CDF ---------------------------------------------------------------------

def test_build_cdf_monotone_and_normalised():
    A = _softmax_weights(4, 32)
    F = build_cdf(A)
    assert F.shape == A.shape
    # non-decreasing along the key dim
    assert torch.all(F[..., 1:] >= F[..., :-1] - 1e-12)
    # last column pinned exactly to 1
    torch.testing.assert_close(F[..., -1], torch.ones(4, dtype=torch.float64))


def test_invert_cdf_recovers_known_indices():
    # Uniform A over n_k=8: F[j] = (j+1)/8. A threshold just above j/8 -> index j.
    n_k = 8
    F = build_cdf(_uniform_weights(1, n_k))
    j = torch.arange(n_k, dtype=torch.float64)
    T = (j + 0.5) / n_k  # midpoint of each mass interval
    idx = invert_cdf(F, T.unsqueeze(0))
    torch.testing.assert_close(idx.squeeze(0), torch.arange(n_k))


def test_invert_cdf_indices_in_range():
    A = _softmax_weights(3, 16)
    F = build_cdf(A)
    T = torch.rand(3, 50, dtype=torch.float64) * (1 - 1e-9)
    idx = invert_cdf(F, T)
    assert idx.min() >= 0 and idx.max() < 16


# --- iid ---------------------------------------------------------------------

def test_iid_shape_range_and_reproducible():
    A = _softmax_weights(4, 32)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = iid_indices(A, 64, generator=g1)
    b = iid_indices(A, 64, generator=g2)
    assert a.shape == (4, 64)
    assert a.min() >= 0 and a.max() < 32
    torch.testing.assert_close(a, b)


# --- stratified --------------------------------------------------------------

def test_stratified_uniform_alignment():
    # Uniform A with S == n_k: stratum m's threshold lies in [m/S,(m+1)/S),
    # which under linear F maps exactly to index m, regardless of the offset.
    n_k = 16
    A = _uniform_weights(2, n_k)
    g = torch.Generator().manual_seed(1)
    idx = stratified_indices(A, n_k, generator=g)
    expected = torch.arange(n_k).expand(2, n_k)
    torch.testing.assert_close(idx, expected)


def test_stratified_one_index_per_stratum():
    # Each draw must fall in its own equal-mass CDF stratum.
    A = _softmax_weights(2, 32)
    F = build_cdf(A)
    S = 8
    g = torch.Generator().manual_seed(2)
    idx = stratified_indices(A, S, generator=g)
    # lower/upper CDF bounds of the returned index must straddle the stratum band
    for h in range(2):
        for m in range(S):
            j = idx[h, m].item()
            lo = F[h, j - 1].item() if j > 0 else 0.0
            hi = F[h, j].item()
            band_lo, band_hi = m / S, (m + 1) / S
            # the index's mass interval [lo,hi) intersects the stratum [band_lo,band_hi)
            assert lo < band_hi and hi > band_lo


# --- systematic --------------------------------------------------------------

def test_systematic_uniform_alignment():
    n_k = 16
    A = _uniform_weights(3, n_k)
    g = torch.Generator().manual_seed(3)
    idx = systematic_indices(A, n_k, generator=g)
    expected = torch.arange(n_k).expand(3, n_k)
    torch.testing.assert_close(idx, expected)


def test_systematic_reproducible_and_shaped():
    A = _softmax_weights(4, 64)
    g1 = torch.Generator().manual_seed(9)
    g2 = torch.Generator().manual_seed(9)
    a = systematic_indices(A, 32, generator=g1)
    b = systematic_indices(A, 32, generator=g2)
    assert a.shape == (4, 32)
    torch.testing.assert_close(a, b)


def test_systematic_one_draw_per_head():
    # Exactly one random number is consumed per head: after drawing the H offsets,
    # the generator state must match having drawn a (H,1) tensor.
    A = _softmax_weights(5, 32)  # float64 weights -> float64 offsets internally
    g = torch.Generator().manual_seed(11)
    systematic_indices(A, 16, generator=g)
    ref = torch.Generator().manual_seed(11)
    torch.rand(5, 1, generator=ref, dtype=torch.float64)  # consume exactly H float64 draws
    # both generators should now produce identical subsequent draws (same dtype)
    torch.testing.assert_close(
        torch.rand(3, generator=g, dtype=torch.float64),
        torch.rand(3, generator=ref, dtype=torch.float64),
    )


# --- unique counts -----------------------------------------------------------

def test_unique_counts_bounded():
    A = _softmax_weights(4, 32)
    g = torch.Generator().manual_seed(5)
    idx = iid_indices(A, 100, generator=g)
    counts = unique_counts(idx)
    assert counts.shape == (4,)
    assert torch.all(counts <= 100) and torch.all(counts <= 32) and torch.all(counts >= 1)


def test_unique_counts_matches_reference():
    # Vectorized count must equal the per-row torch.unique reference, incl. edges.
    A = _softmax_weights(6, 20)
    g = torch.Generator().manual_seed(8)
    idx = iid_indices(A, 37, generator=g)
    ref = torch.tensor([int(idx[h].unique().numel()) for h in range(idx.shape[0])])
    torch.testing.assert_close(unique_counts(idx), ref)
    # empty draw -> zeros
    empty = torch.empty(3, 0, dtype=torch.long)
    assert torch.equal(unique_counts(empty), torch.zeros(3, dtype=torch.long))
