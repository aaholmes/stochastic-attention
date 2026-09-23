"""Where does the hybrid's crack open? Read-matched variance vs concentration.

The real-model perplexity sweep showed plain ``santa_sys`` beats ``santa_hybrid`` on the *bytes* frontier
because, when attention is concentrated, sys's with-replacement collisions buy many
samples per unique read — leverage the deterministic head can't match. The open
question: is there a *diffuse* regime where collisions vanish and the
head's guarantee wins?

This answers it on synthetic tensors (no model): sweep attention concentration via a
temperature knob, and for each level compare the read-matched frontiers — variance
vs **expected unique-read fraction** — of sys and hybrid(k_h=1). Crossover =
the concentration where hybrid's frontier drops below sys's.
"""

from __future__ import annotations

import numpy as np
import torch

from ..attn import attn
from ..attn.geometry import attn_weights
from .variance import variance_trace

BUDGETS = [4, 8, 16, 32, 64, 128, 256]
SCALES = [0.2, 0.4, 0.7, 1.0, 2.0, 4.0]  # q-scale: lower = flatter/diffuse, higher = peaked


def _qkv(scale: float, *, n_k: int, H: int, d: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(H, d, generator=g, dtype=torch.float64) * scale
    K = torch.randn(n_k, H, d, generator=g, dtype=torch.float64)  # H_kv=H here (no GQA)
    V = torch.randn(n_k, H, d, generator=g, dtype=torch.float64)
    return q, K, V


def participation_ratio(q, K) -> float:
    A = attn_weights(q, K)  # [H, n_k]
    return float((1.0 / A.pow(2).sum(-1)).mean())


def _measure(impl, q, K, V, *, runs, base_seed, **cfg) -> tuple[float, float]:
    """Return (mean expected-unique read fraction, variance-trace) for one config."""
    n_k = K.shape[0]
    k_h = int(cfg.get("k_h", 0))
    outs, read_fracs = [], []
    for r in range(runs):
        gen = torch.Generator().manual_seed(base_seed + r)
        out, info = attn(q, K, V, impl=impl, generator=gen, return_info=True, **cfg)
        outs.append(out.to(torch.float64))
        reads = (min(k_h, n_k) + info.unique.to(torch.float64)).clamp_(max=float(n_k))
        read_fracs.append(float((reads / n_k).mean()))
    return float(np.mean(read_fracs)), variance_trace(torch.stack(outs))


def build_crossover(*, n_k=512, H=8, d=16, runs=400, seed=0) -> list[dict]:
    """For each concentration level, the sys and hybrid(k_h=1) read/variance frontiers."""
    rows = []
    for scale in SCALES:
        q, K, V = _qkv(scale, n_k=n_k, H=H, d=d, seed=seed)
        sys = [(*_measure("santa_sys", q, K, V, runs=runs, base_seed=seed, S=S),) for S in BUDGETS]
        hyb = [(*_measure("santa_hybrid", q, K, V, runs=runs, base_seed=seed, k_h=1, S=S),)
               for S in BUDGETS]
        rows.append({
            "scale": scale, "participation_ratio": participation_ratio(q, K),
            "sys": sys, "hybrid_kh1": hyb,
        })
    return rows


def _ratios(row: dict) -> list[float]:
    """var_hybrid / sys_var(at the same read fraction) at each hybrid point. <1 ⇒ hybrid lower."""
    sx = np.log([rf for rf, _ in row["sys"]])
    sy = np.log([v for _, v in row["sys"]])
    order = np.argsort(sx)
    sx, sy = sx[order], sy[order]
    out = []
    for rf, v in row["hybrid_kh1"]:
        sys_v = np.exp(np.interp(np.log(rf), sx, sy))
        out.append(v / sys_v)
    return out


def _hybrid_advantage(row: dict) -> float:
    """Best (minimum) variance ratio across hybrid points. <1 ⇒ hybrid wins somewhere."""
    return float(min(_ratios(row)))


def _typical_advantage(row: dict) -> float:
    """Geometric-mean variance ratio across the frontier — the *fair* summary."""
    return float(np.exp(np.mean(np.log(_ratios(row)))))


def plot_crossover(rows: list[dict], out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
    for i, row in enumerate(rows):
        ax = axes[i // ncol][i % ncol]
        sx = [100 * rf for rf, _ in row["sys"]]
        sy = [v for _, v in row["sys"]]
        hx = [100 * rf for rf, _ in row["hybrid_kh1"]]
        hy = [v for _, v in row["hybrid_kh1"]]
        ax.loglog(sx, sy, "-o", color="tab:green", label="sys", markersize=4)
        ax.loglog(hx, hy, "-s", color="tab:purple", label="hybrid k_h=1", markersize=4)
        ax.set_title(f"participation ratio ≈ {row['participation_ratio']:.0f}", fontsize=9)
        ax.set_xlabel("rows read (%)", fontsize=8)
        ax.set_ylabel("variance", fontsize=8)
        ax.grid(True, which="major", ls=":", alpha=0.3)
        ax.legend(fontsize=7, frameon=False)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    import json
    from pathlib import Path

    from .stamp import stamp

    rows = build_crossover()
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    payload = stamp({"kind": "crossover", "budgets": BUDGETS, "rows": rows})
    sha = payload.get("git_sha", "nogit")[:8]
    png = out_dir / f"crossover_{sha}.png"
    plot_crossover(rows, png)
    (out_dir / f"crossover_{sha}.json").write_text(json.dumps(payload, indent=2))

    print(f"{'PR (eff. tokens)':>16} {'typical h/sys':>14} {'best h/sys':>12}  verdict")
    for row in rows:
        typ, best = _typical_advantage(row), _hybrid_advantage(row)
        verdict = "~tie" if 0.9 <= typ <= 1.1 else ("hybrid" if typ < 1 else "sys")
        print(f"{row['participation_ratio']:>16.1f} {typ:>14.2f} {best:>12.2f}  {verdict}")
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
