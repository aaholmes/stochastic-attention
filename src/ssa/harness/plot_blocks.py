"""Variance vs contiguous-block size B at a fixed read budget (design doc §0.5).

At a fixed nominal read budget R, sampling larger blocks means fewer block draws
(S = R/B) but each block contributes its *exact* within-block mean. This traces the
trade-off: how does estimator variance move as you coarsen from row-level (B=1, pure
sampling) toward whole-cache (B→n_k, exact/dense)? The achieved read fraction is
reported alongside, since block collisions make it < nominal.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ..attn import attn
from .stamp import stamp
from .variance import variance_trace

B_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]


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


def plot_block_curve(rows, out_path, *, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Bs = [r["B"] for r in rows]
    var = [r["variance"] for r in rows]
    rf = [100 * r["read_fraction"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.loglog(Bs, var, "-o", color="tab:blue", label="variance-trace")
    for r, b, v in zip(rows, Bs, var):
        ax.annotate(f"{100*r['read_fraction']:.1f}%", (b, v), fontsize=7,
                    xytext=(3, 4), textcoords="offset points")
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
    args = p.parse_args()

    q, K, V = _qkv(args.scale, args.n_k, args.seed)
    rows = build_block_curve(q, K, V, read_budget=args.read_budget, method=args.method,
                             runs=args.runs, base_seed=args.seed)

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    payload = stamp({"kind": "variance_vs_block", "read_budget": args.read_budget,
                     "n_k": args.n_k, "q_scale": args.scale, "method": args.method, "rows": rows})
    sha = payload.get("git_sha", "nogit")[:8]
    png = out_dir / f"variance_vs_block_{sha}.png"
    plot_block_curve(rows, png, title=f"Variance vs block size (budget={args.read_budget}, "
                                      f"n_k={args.n_k}, q_scale={args.scale})")
    (out_dir / f"variance_vs_block_{sha}.json").write_text(json.dumps(payload, indent=2))

    print(f"{'B':>5} {'S':>5} {'read%':>7} {'variance':>12}")
    for r in rows:
        print(f"{r['B']:>5} {r['S']:>5} {100*r['read_fraction']:>6.1f}% {r['variance']:>12.4e}")
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
