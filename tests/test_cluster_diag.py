"""Tests for the cluster diagnostic (k-means, moments, free-energy, pruning)."""

from __future__ import annotations

import torch

from ssa.harness.cluster_diag import (
    cluster_stats,
    kmeans,
    query_metrics,
    summarize,
)


def _blobs(n_per, centers, spread, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = [c + spread * torch.randn(n_per, len(c), generator=g) for c in centers]
    return torch.cat(pts, 0)


def test_kmeans_recovers_separated_blobs():
    centers = [torch.tensor([0.0, 0.0]), torch.tensor([10.0, 10.0]), torch.tensor([0.0, 10.0])]
    X = _blobs(40, centers, spread=0.2, seed=1)
    labels, c = kmeans(X, 3, seed=0)
    # each true blob should be (almost) pure in one cluster
    for b in range(3):
        block = labels[b * 40:(b + 1) * 40]
        dominant = block.bincount().max().item()
        assert dominant >= 38


def test_radius_bound_is_a_true_upper_bound():
    X = _blobs(20, [torch.tensor([0.0, 0.0]), torch.tensor([5.0, 0.0])], 0.5, seed=2)
    labels, _ = kmeans(X, 2, seed=0)
    stats = cluster_stats(X, labels, 2)
    q = torch.tensor([1.0, 2.0], dtype=torch.float64)
    m = query_metrics(q, X, labels, stats)
    assert m["upper_ok"]  # |b|·e^{q·c+‖q‖R} ≥ true mass, always


def test_clustering_beats_random_on_pruning_and_free_energy():
    # Keys cluster into tight blobs; a query aligned with one blob.
    d = 8
    g = torch.Generator().manual_seed(3)
    blob_centers = [5.0 * torch.randn(d, generator=g) for _ in range(8)]
    K = _blobs(16, blob_centers, spread=0.3, seed=4)
    q = (blob_centers[0] / blob_centers[0].norm()).to(torch.float64)  # points at blob 0
    Q = q.unsqueeze(0)

    s = summarize(K, Q, B=16, target=0.99, seed=0)
    km, rnd = s["kmeans"], s["random"]
    # tight content clusters => fetch far fewer keys for 99% mass than random blocks
    assert km["frac_keys_for_99pct_mass"] < rnd["frac_keys_for_99pct_mass"]
    # and the Gaussian free-energy estimate is good on coherent clusters
    assert km["free_energy_err_nats"] < rnd["free_energy_err_nats"]
    assert km["radius_bound_valid"]


def test_summarize_shapes_and_schemes():
    K = _blobs(8, [torch.zeros(4), torch.ones(4) * 3], 0.5, seed=5)
    Q = torch.randn(5, 4, dtype=torch.float64)
    s = summarize(K, Q, B=8)
    assert s["n_k"] == 16 and s["n_queries"] == 5
    for scheme in ("kmeans", "contiguous", "random"):
        assert 0.0 < s[scheme]["frac_keys_for_99pct_mass"] <= 1.0
