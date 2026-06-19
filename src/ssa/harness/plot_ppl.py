"""Plot the Phase C frontier: perplexity increase vs value-read fraction.

Reads one or more stamped ``phaseC_ppl_*.json`` files (from ``ssa.harness.ppl_sweep``)
and overlays them, so a focused run (e.g. the cheap-end hybrids + top-k) can be
drawn against an earlier sweep's ``santa_sys`` curve without re-running it.

Lower-left is better (fewer value rows read, smaller quality hit). ``santa_sys``
and ``topk`` are swept curves; ``santa_hybrid`` is grouped by head size ``k_h``.

Run:
    uv run python -m ssa.harness.plot_ppl A.json [B.json ...] --out frontier.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_results(paths) -> tuple[list[dict], dict]:
    """Merge results from several JSONs, de-duping by (impl, cfg)."""
    merged: dict[tuple, dict] = {}
    first_payload = None
    for p in paths:
        payload = json.loads(Path(p).read_text())
        first_payload = first_payload or payload
        for r in payload["results"]:
            key = (r["impl"], tuple(sorted(r.get("cfg", {}).items())))
            merged.setdefault(key, r)
    return list(merged.values()), first_payload


def _delta_pct(r: dict, base: float) -> float:
    return 100.0 * (r["ppl_mean"] - base) / base


def plot_ppl_frontier(results: list[dict], out_path, *, title: str, ymax: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in results if "error" not in r]
    dense = next(r for r in ok if r["impl"] == "dense")
    base = dense["ppl_mean"]

    def xy(rs):
        rs = sorted(rs, key=lambda r: r["read_fraction"])
        return [100 * r["read_fraction"] for r in rs], [_delta_pct(r, base) for r in rs], rs

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # santa_sys — baseline sampling curve
    xs, ys, rs = xy(r for r in ok if r["impl"] == "santa_sys")
    if xs:
        ax.plot(xs, ys, "-o", color="tab:green", label="santa_sys", zorder=3)
        for r, x, y in zip(rs, xs, ys):
            ax.annotate(f"{r['cfg']['S']}", (x, y), fontsize=6, xytext=(3, 4),
                        textcoords="offset points", color="tab:green")

    # topk — biased deterministic baseline
    xs, ys, rs = xy(r for r in ok if r["impl"] == "topk")
    if xs:
        ax.plot(xs, ys, "--x", color="tab:orange", label="topk (biased)", zorder=3)

    # santa_hybrid — one line per head size k_h (shows det:stoch ratio effect)
    hyb = [r for r in ok if r["impl"] == "santa_hybrid"]
    k_hs = sorted({r["cfg"]["k_h"] for r in hyb})
    cmap = plt.get_cmap("viridis")
    for i, kh in enumerate(k_hs):
        xs, ys, rs = xy(r for r in hyb if r["cfg"]["k_h"] == kh)
        color = cmap(i / max(len(k_hs) - 1, 1))
        ax.plot(xs, ys, "-s", color=color, markersize=5, label=f"hybrid k_h={kh}", zorder=4)

    ax.axhline(0.0, color="gray", ls=":", lw=1, label="dense (no quality loss)")
    if ymax is not None:
        ax.set_ylim(top=ymax)
    ax.set_xscale("log")
    ax.set_xlabel("value rows read (% of cache)  —  lower is cheaper")
    ax.set_ylabel("perplexity increase vs dense (%)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results_json", nargs="+")
    p.add_argument("--out", default=None, help="PNG path (default: alongside the first JSON)")
    p.add_argument("--title", default=None)
    p.add_argument("--ymax", type=float, default=None, help="clip the y-axis for readability")
    args = p.parse_args()

    results, payload = load_results(args.results_json)
    src = Path(args.results_json[0])
    out = Path(args.out) if args.out else src.with_suffix(".png")
    title = args.title or (
        f"PPL vs value-read fraction — {payload.get('model', '?')}, "
        f"ctx={payload.get('chunk_len', '?')}"
    )
    plot_ppl_frontier(results, out, title=title, ymax=args.ymax)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
