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


def test_query_whitening_ignores_irrelevant_spread():
    # Keys vary hugely along a dimension no query ever points at; content lives in dims 0-1.
    g = torch.Generator().manual_seed(7)
    n, d = 64, 4
    content = torch.cat([  # two content blobs in dims 0-1
        torch.tensor([0.0, 0.0]) + 0.2 * torch.randn(n // 2, 2, generator=g),
        torch.tensor([4.0, 0.0]) + 0.2 * torch.randn(n // 2, 2, generator=g),
    ])
    noise = 50.0 * torch.randn(n, d - 2, generator=g)        # giant query-irrelevant spread
    K = torch.cat([content, noise], dim=1)
    Q = torch.randn(40, d, generator=g)
    Q[:, 2:] = 0.0                                            # queries live only in the content dims

    s = summarize(K, Q, B=8)
    # Euclidean k-means is dominated by the noise dims; query-whitening clusters by content,
    # so its free-energy estimate and pruning are no worse — and typically better.
    assert s["kmeans_qw"]["free_energy_err_nats"] <= s["kmeans"]["free_energy_err_nats"] + 1e-6
    assert s["kmeans_qw"]["frac_keys_for_99pct_mass"] <= s["kmeans"]["frac_keys_for_99pct_mass"] + 1e-6


def test_within_cluster_participation_ratio():
    from ssa.harness.cluster_diag import cluster_stats, query_metrics

    # two clusters of 4 keys each (labels 0,0,0,0,1,1,1,1)
    K = torch.eye(8)[:, :8].double()  # orthonormal keys so scores = q components
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    stats = cluster_stats(K, labels, 2)

    # query makes cluster 0 uniformly hot (4 equal scores) and cluster 1 single-peaked
    q = torch.tensor([2.0, 2.0, 2.0, 2.0, 9.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    m = query_metrics(q, K, labels, stats)
    pr = m["within_pr"]
    assert pr[0] > 3.5            # uniform cluster: ~4 tokens carry the mass
    assert pr[1] < 1.2            # peaked cluster: ~1 token dominates
    # mass-weighted is dominated by the hot (peaked) cluster -> near 1
    assert m["within_pr_massw"] < 1.5


def test_summarize_shapes_and_schemes():
    K = _blobs(8, [torch.zeros(4), torch.ones(4) * 3], 0.5, seed=5)
    Q = torch.randn(5, 4, dtype=torch.float64)
    s = summarize(K, Q, B=8)
    assert s["n_k"] == 16 and s["n_queries"] == 5
    for scheme in ("kmeans", "contiguous", "random"):
        assert 0.0 < s[scheme]["frac_keys_for_99pct_mass"] <= 1.0
