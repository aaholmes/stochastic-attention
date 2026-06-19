"""Variance convergence vs total sample budget (the Phase A/B "money figure").

For the plain samplers the x-axis is the sample count ``S``. For ``santa_hybrid``
the budget counts **both** halves: the deterministic head and the stochastic tail,
``total = k_h + S_tail`` — that is the read budget the estimators actually spend.
Plotting variance-trace vs that total on log-log axes shows semi-stochastic
estimators converging faster (lower variance at equal budget) than plain sampling.

Run:
    uv run python -m ssa.harness.plot_variance --scale 4 --runs 800
writes a PNG + stamped JSON to ``src/ssa/results/``.
"""

from __future__ import annotations

import argparse

import torch

from .stamp import stamp
from .variance import collect_estimates, fit_slope, variance_trace

DEFAULT_BUDGETS = [8, 16, 32, 64, 128, 256]


def build_curves(
    q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    budgets: list[int] = DEFAULT_BUDGETS,
    hybrid_k_h: tuple[int, ...] = (4, 16),
    runs: int = 800,
    base_seed: int = 0,
) -> dict:
    """Variance-trace vs total sample budget for each estimator/parameter set.

    Returns ``{label: {"x": [...], "var": [...], "slope": float}}``. For hybrid
    curves ``x`` is the total budget ``k_h + S_tail``; a point is only included
    when at least one tail sample fits (``total > k_h``).
    """
    curves: dict[str, dict] = {}

    for impl in ("santa", "santa_strat", "santa_sys"):
        var = [
            variance_trace(collect_estimates(impl, q, K, V, S=b, runs=runs, base_seed=base_seed))
            for b in budgets
        ]
        curves[impl] = {"x": list(budgets), "var": var, "slope": fit_slope(budgets, var)}

    for k_h in hybrid_k_h:
        xs = [b for b in budgets if b > k_h]
        var = [
            variance_trace(
                collect_estimates(
                    "santa_hybrid", q, K, V, S=b - k_h, runs=runs, base_seed=base_seed,
                    k_h=k_h, tail="sys",
                )
            )
            for b in xs
        ]
        curves[f"santa_hybrid(k_h={k_h})"] = {
            "x": xs, "var": var, "slope": fit_slope(xs, var) if len(xs) > 1 else float("nan"),
        }

    return curves


def plot_curves(curves: dict, path, *, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, c in curves.items():
        ax.loglog(c["x"], c["var"], marker="o", label=f"{label}  (slope {c['slope']:+.2f})")
    ax.set_xlabel("total samples  (hybrid: $k_h + S_{tail}$)")
    ax.set_ylabel("variance-trace of estimator")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _qkv(scale: float, n_k: int, seed: int):
    H, H_kv, d = 8, 2, 16
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(H, d, generator=gen, dtype=torch.float64) * scale
    K = torch.randn(n_k, H_kv, d, generator=gen, dtype=torch.float64)
    V = torch.randn(n_k, H_kv, d, generator=gen, dtype=torch.float64)
    return {"H": H, "H_kv": H_kv, "d": d, "n_k": n_k, "q_scale": scale}, q, K, V


def main() -> None:
    import json
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=float, default=4.0, help="q scale; higher = more concentrated A")
    p.add_argument("--n-k", type=int, default=256)
    p.add_argument("--runs", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    geom, q, K, V = _qkv(args.scale, args.n_k, args.seed)
    curves = build_curves(q, K, V, runs=args.runs, base_seed=args.seed)

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    payload = stamp({"phase": "A/B", "kind": "variance_convergence",
                     "geometry": geom, "runs": args.runs, "curves": curves})
    sha = payload.get("git_sha", "nogit")[:8]
    png = out_dir / f"variance_convergence_{sha}.png"
    js = out_dir / f"variance_convergence_{sha}.json"
    plot_curves(curves, png, title=f"Variance convergence (q_scale={args.scale}, n_k={args.n_k})")
    js.write_text(json.dumps(payload, indent=2))

    for label, c in curves.items():
        print(f"{label:26s} slope={c['slope']:+.3f}")
    print(f"\nwrote {png}\n      {js}")


if __name__ == "__main__":
    main()
