"""Plot the bias / variance split of stochastic-attention decode vs read budget S,
plain vs debias-LoRA. Shows what the single-draw acceptance metric couldn't:
whether the debiaser removes the *systematic* Jensen-gap bias (the correctable part)
as opposed to the per-draw variance (which no deterministic LoRA can touch).

Reads bias_<corpus>_plain.json (multi-S) and bias_<corpus>_debiased_s<S>.json (per-S).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("src/ssa/results")


def load(corpus):
    plain = {r["S"]: r for r in json.loads((R / f"bias_{corpus}_plain.json").read_text())["results"]}
    deb = {}
    for f in sorted(R.glob(f"bias_{corpus}_debiased_s*.json")):
        rows = json.loads(f.read_text())["results"]
        for r in rows:
            deb[r["S"]] = r
    return plain, deb


def main():
    corpus = "code"
    plain, deb = load(corpus)
    S = sorted(plain)
    x = range(len(S))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    # Plain: stacked bias (dark) + variance (light).
    pb = [plain[s]["bias"] for s in S]
    pv = [plain[s]["variance"] for s in S]
    ax.bar([i - w / 2 for i in x], pb, w, color="tab:red", label="plain — bias (systematic)")
    ax.bar([i - w / 2 for i in x], pv, w, bottom=pb, color="mistyrose",
           label="plain — variance (per-draw)")
    # Debiased: only where a LoRA exists.
    for i, s in enumerate(S):
        if s not in deb:
            continue
        db, dv = deb[s]["bias"], deb[s]["variance"]
        ax.bar(i + w / 2, db, w, color="tab:blue", label="debiased — bias" if i == 0 else None)
        ax.bar(i + w / 2, dv, w, bottom=db, color="lightsteelblue",
               label="debiased — variance" if i == 0 else None)
        # annotate bias reduction
        ax.annotate(f"{100*(1-db/pb[i]):+.0f}%\nbias", (i + w / 2, db), ha="center",
                    va="bottom", fontsize=8, color="tab:blue")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"S={s}\n({100*plain[s]['read_fraction']:.0f}% reads)" for s in S])
    ax.set_ylabel("TVD to dense (per token)", fontsize=11)
    ax.set_title(f"Stochastic-attention error: systematic bias vs per-draw variance "
                 f"({corpus}, Qwen3-4B)\nleft = plain, right = debias-LoRA; "
                 f"a working debiaser shrinks the dark (bias) block", fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = R / "bias_split_plot.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
