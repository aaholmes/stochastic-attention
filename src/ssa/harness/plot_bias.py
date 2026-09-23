"""Plot the bias / variance split of sampled-attention decode vs sample budget S.

The error of the sampled model's next-token distribution, measured as total
variation distance (TVD) to the dense model, splits into a systematic part that
survives averaging over draws (the Jensen-gap bias) and a per-draw variance part.
Per S: grouped stacked bars [plain | single-draw LoRA | multi-draw LoRA], each split
bias (dark) + variance (light). LoRA is a low-rank adapter trained by
``ssa.harness.debias_train`` to reduce that TVD.

The ``read_fraction`` field in the bias JSONs is a known recording bug (it reads 1.0
although sampling did happen), so the x-axis is labelled by S alone.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("src/ssa/results")


def _rows(path):
    return {r["S"]: r for r in json.loads((R / path).read_text())["results"]}


def main():
    plain = _rows("bias_code_plain.json")
    single = {}
    md = {}
    for f in sorted(R.glob("bias_code_debiased_s*.json")):
        S = json.loads(f.read_text())["results"][0]["S"]
        (md if f.stem.endswith("_md") else single)[S] = json.loads(f.read_text())["results"][0]

    S = sorted(plain)
    x = range(len(S))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 5))

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

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in S])
    ax.set_xlabel("samples per head per step, S")
    ax.set_ylabel("TVD to dense model (mean per token)")
    meta = json.loads((R / "bias_code_plain.json").read_text())
    ax.text(0.99, 0.97, f"{meta['model']}, {meta['corpus']} corpus, "
            f"n = {plain[S[0]]['n_steps']} decode steps × {meta['draws']} draws",
            transform=ax.transAxes, fontsize=8, color="0.3", va="top", ha="right")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    out = R / "bias_split_plot.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
