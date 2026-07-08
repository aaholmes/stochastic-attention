"""Plot the bias / variance split of stochastic-attention decode vs read budget S.

Two findings in one figure: (1) the systematic Jensen-gap bias (dark) dominates the
per-draw variance (light) at the low read budgets that matter, scaling ~1/S; (2) the
TVD-trained LoRA — single-draw OR multi-draw-averaged — barely dents the bias, so the
correctable-in-principle bias is not in a rank-16 adapter's reach.

Per S: grouped stacked bars [plain | single-draw LoRA | multi-draw LoRA], each split
bias (dark) + variance (light). Read-fractions come from accept_sweep_code.json (the
`read_fraction` field in the bias json is a known display bug — sampling still happened).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("src/ssa/results")


def _rows(path):
    return {r["S"]: r for r in json.loads((R / path).read_text())["results"]}


def _reads_map():
    try:
        rs = json.loads((R / "accept_sweep_code.json").read_text())["results"]
        return {r["S"]: r["read_fraction"] for r in rs}
    except Exception:
        return {}


def main():
    plain = _rows("bias_code_plain.json")
    single = {}
    md = {}
    for f in sorted(R.glob("bias_code_debiased_s*.json")):
        S = json.loads(f.read_text())["results"][0]["S"]
        (md if f.stem.endswith("_md") else single)[S] = json.loads(f.read_text())["results"][0]
    reads = _reads_map()

    S = sorted(plain)
    x = range(len(S))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.2))

    def stacked(series, offset, cbias, cvar, label):
        xs, bs, vs = [], [], []
        for i, s in enumerate(S):
            if s in series:
                xs.append(i + offset); bs.append(series[s]["bias"]); vs.append(series[s]["variance"])
        ax.bar(xs, bs, w, color=cbias, label=f"{label} — bias")
        ax.bar(xs, vs, w, bottom=bs, color=cvar, label=f"{label} — variance")

    stacked(plain, -w, "tab:red", "mistyrose", "plain")
    stacked(single, 0.0, "tab:blue", "lightsteelblue", "single-draw LoRA")
    stacked(md, +w, "tab:green", "honeydew", "multi-draw LoRA")

    # bias-fraction callout on the plain bar
    for i, s in enumerate(S):
        frac = plain[s]["bias"] / plain[s]["per_draw_tvd"]
        ax.annotate(f"{100*frac:.0f}%\nbias", (i - w, plain[s]["bias"] + plain[s]["variance"]),
                    ha="center", va="bottom", fontsize=8, color="tab:red")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"S={s}\n({100*reads.get(s, float('nan')):.1f}% reads)" for s in S])
    ax.set_ylabel("TVD to dense model (per token)", fontsize=11)
    ax.set_title("Stochastic-attention error: systematic bias vs per-draw variance (code, Qwen3-4B)\n"
                 "bias dominates at low read budgets and scales ~1/S; the TVD-LoRA barely removes it",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = R / "bias_split_plot.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
