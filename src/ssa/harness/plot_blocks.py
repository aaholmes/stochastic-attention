"""Variance vs contiguous-block size B at a fixed read budget (design doc §0.5).

At a fixed nominal read budget R, sampling larger blocks means fewer block draws
(S = R/B) but each block contributes its *exact* within-block mean. This traces the
trade-off: how does estimator variance move as you coarsen from row-level (B=1, pure
sampling) toward whole-cache (B→n_k, exact/dense)? The achieved read fraction is
reported alongside, since block collisions make it < nominal.

**Content-clustered layout (the actual `santa_block` premise).** Contiguous-block
sampling only pays off if a block's rows are *all* useful — i.e. attention mass
clusters contiguously. On the native (arrival-order) key layout it does not, so
larger blocks read cold neighbours and lose to row-level sampling. This harness
therefore also measures a **clustered layout**: cluster each kv-head's keys by
direction and permute the cache so similar keys sit adjacent, then sample blocks
over that reordering. A permutation of the key axis cannot bias `santa_block` (it is
unbiased for *any* partition, design doc §0.5), so this is a pure efficiency test —
does clustering make a hot block uniformly hot, and does that beat native-order
block sampling on variance at a fixed read budget?
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ..attn import attn
from .cluster_diag import kmeans
from .stamp import stamp
from .variance import variance_trace

B_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]


def cluster_permutation(K: torch.Tensor, n_clusters: int, *, seed: int = 0) -> torch.Tensor:
    """Per-kv-head permutation ordering keys so same-cluster keys are contiguous.

    ``K``: ``[n_k, H_kv, d]``. Clusters each kv-head's *directions* (unit-normalised
    keys — content, not magnitude) with Lloyd k-means, then orders positions by
    cluster label. Returns ``perm`` ``[H_kv, n_k]`` where ``perm[h]`` reorders that
    head's key axis. Each kv-head is a physically separate cache region, so heads may
    carry independent orderings; the ``G`` query heads sharing a kv-head inherit it.
    """
    n_k, H_kv, _ = K.shape
    k = max(1, min(n_clusters, n_k))
    perms = []
    for h in range(H_kv):
        Kh = K[:, h, :].to(torch.float64)
        Xh = Kh / Kh.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        labels, _ = kmeans(Xh, k, seed=seed)
        perms.append(torch.argsort(labels, stable=True))
    return torch.stack(perms)


def apply_permutation(T: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Apply a per-kv-head permutation ``[H_kv, n_k]`` to a ``[n_k, H_kv, d]`` cache."""
    out = torch.empty_like(T)
    for h in range(perm.shape[0]):
        out[:, h, :] = T[perm[h], h, :]
    return out


def _measure_block(q, K, V, *, S, B, method, runs, base_seed):
    """(achieved read fraction, variance_trace) for santa_block at this (S, B)."""
    n_k = K.shape[0]
    outs, reads = [], []
    for r in range(runs):
        gen = torch.Generator(device=q.device).manual_seed(base_seed + r)
        out, info = attn(q, K, V, impl="santa_block", S=S, B=B, method=method,
                         generator=gen, return_info=True)
        outs.append(out.to(torch.float64))
        reads.append(float((info.reads.to(torch.float64) / n_k).mean()))
    return float(np.mean(reads)), variance_trace(torch.stack(outs))


def build_block_curve(q, K, V, *, read_budget, B_values=B_VALUES, method="sys",
                      runs=600, base_seed=0) -> list[dict]:
    """For each B, S = max(1, read_budget // B); measure variance + achieved reads."""
    rows = []
    for B in B_values:
        S = max(1, read_budget // B)
        rf, var = _measure_block(q, K, V, S=S, B=B, method=method, runs=runs, base_seed=base_seed)
        rows.append({"B": B, "S": S, "read_fraction": rf, "variance": var})
    return rows


def build_layout_curves(q, K, V, *, read_budget, n_clusters, B_values=B_VALUES,
                        method="sys", runs=600, base_seed=0, seed=0):
    """Native-order vs content-clustered block curves on the same (q, K, V).

    Clustering permutes each kv-head's key axis so similar keys are contiguous, then
    reuses the identical block sweep. Returns ``(native_rows, cluster_rows)``.
    """
    native = build_block_curve(q, K, V, read_budget=read_budget, B_values=B_values,
                               method=method, runs=runs, base_seed=base_seed)
    perm = cluster_permutation(K, n_clusters, seed=seed)
    K_c, V_c = apply_permutation(K, perm), apply_permutation(V, perm)
    cluster = build_block_curve(q, K_c, V_c, read_budget=read_budget, B_values=B_values,
                                method=method, runs=runs, base_seed=base_seed)
    return native, cluster


def _annotate_reads(ax, rows, color):
    for r in rows:
        ax.annotate(f"{100*r['read_fraction']:.1f}%", (r["B"], r["variance"]), fontsize=7,
                    color=color, xytext=(3, 4), textcoords="offset points")


def plot_block_curve(rows, out_path, *, title: str, cluster_rows=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Bs = [r["B"] for r in rows]
    var = [r["variance"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.loglog(Bs, var, "-o", color="tab:blue", label="native order")
    _annotate_reads(ax, rows, "tab:blue")
    if cluster_rows is not None:
        cvar = [r["variance"] for r in cluster_rows]
        ax.loglog([r["B"] for r in cluster_rows], cvar, "-s", color="tab:red",
                  label="content-clustered")
        _annotate_reads(ax, cluster_rows, "tab:red")
    ax.set_xlabel("block size B  (B=1: row-level santa  →  B≥n_k: exact/dense)")
    ax.set_ylabel("variance-trace at fixed read budget")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _qkv(scale, n_k, seed):
    H, H_kv, d = 8, 2, 16
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(H, d, generator=g, dtype=torch.float64) * scale
    K = torch.randn(n_k, H_kv, d, generator=g, dtype=torch.float64)
    V = torch.randn(n_k, H_kv, d, generator=g, dtype=torch.float64)
    return q, K, V


def main() -> None:
    import json
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--read-budget", type=int, default=128)
    p.add_argument("--scale", type=float, default=4.0)
    p.add_argument("--n-k", type=int, default=512)
    p.add_argument("--method", default="sys")
    p.add_argument("--runs", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clusters", type=int, default=None,
                   help="k-means cluster count for the content-clustered layout "
                        "(default: n_k // 16). Set 0 to skip the clustered curve.")
    args = p.parse_args()

    q, K, V = _qkv(args.scale, args.n_k, args.seed)
    with_cluster = args.clusters != 0
    n_clusters = args.clusters if args.clusters else max(1, args.n_k // 16)

    if with_cluster:
        rows, cluster_rows = build_layout_curves(
            q, K, V, read_budget=args.read_budget, n_clusters=n_clusters,
            method=args.method, runs=args.runs, base_seed=args.seed, seed=args.seed)
    else:
        rows = build_block_curve(q, K, V, read_budget=args.read_budget, method=args.method,
                                 runs=args.runs, base_seed=args.seed)
        cluster_rows = None

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    payload = stamp({"kind": "variance_vs_block", "read_budget": args.read_budget,
                     "n_k": args.n_k, "q_scale": args.scale, "method": args.method,
                     "n_clusters": n_clusters if with_cluster else None,
                     "rows": rows, "cluster_rows": cluster_rows})
    sha = payload.get("git_sha", "nogit")[:8]
    png = out_dir / f"variance_vs_block_{sha}.png"
    plot_block_curve(rows, png, cluster_rows=cluster_rows,
                     title=f"Variance vs block size (budget={args.read_budget}, "
                           f"n_k={args.n_k}, q_scale={args.scale})")
    (out_dir / f"variance_vs_block_{sha}.json").write_text(json.dumps(payload, indent=2))

    hdr = f"{'B':>5} {'S':>5} {'read%':>7} {'variance':>12}"
    print("native order\n" + hdr)
    for r in rows:
        print(f"{r['B']:>5} {r['S']:>5} {100*r['read_fraction']:>6.1f}% {r['variance']:>12.4e}")
    if cluster_rows is not None:
        print(f"\ncontent-clustered (k={n_clusters})\n" + hdr)
        for r in cluster_rows:
            print(f"{r['B']:>5} {r['S']:>5} {100*r['read_fraction']:>6.1f}% {r['variance']:>12.4e}")
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
