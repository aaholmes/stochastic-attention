"""Attention-concentration diagnostic (why sys's collisions help, measured).

Collisions in with-replacement sampling are governed by how *concentrated* the
attention distribution is — not by context length. This records, per head/layer
over real decode steps, the entropy and participation ratio of ``A = softmax(qKᵀ/√d)``,
the cumulative mass of the top-k tokens, and — most importantly — the **expected
unique-read fraction at a given sample budget**:

    E[#unique | S draws] / n_k  =  (1/n_k) Σ_j [1 − (1 − A_j)^S]

which is exactly the collision/leverage curve that determines whether the
deterministic head can ever beat plain sampling on the bytes axis.

The diagnostic op computes *dense* attention (exact), so it doesn't perturb the
model — it only observes `A`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from engine.attention import Attention

from ..attn.geometry import attn_weights, gqa_expand, weighted_sum

TOP_KS = (1, 4, 8, 16, 32)
SAMPLE_BUDGETS = (8, 32, 128)
_PR_BIN_EDGES = [2.0 ** i for i in range(0, 13)]  # 1 .. 4096


@dataclass
class ConcentrationStats:
    n: int = 0
    entropy: float = 0.0          # Σ over records of mean-over-heads entropy (nats)
    pr: float = 0.0               # participation ratio 1/Σ A²  (effective # tokens)
    nk: float = 0.0
    top_mass: dict = field(default_factory=lambda: {k: 0.0 for k in TOP_KS})
    uniq_frac: dict = field(default_factory=lambda: {s: 0.0 for s in SAMPLE_BUDGETS})
    pr_hist: list = field(default_factory=lambda: [0] * (len(_PR_BIN_EDGES) - 1))

    @torch.no_grad()
    def record(self, A: torch.Tensor) -> None:
        """A: [H, n_k] attention weights for one (layer, decode step)."""
        Af = A.to(torch.float64)
        H, n_k = Af.shape
        self.n += 1
        self.nk += n_k

        ent = -(Af.clamp_min(1e-300) * Af.clamp_min(1e-300).log()).sum(-1)   # [H]
        self.entropy += float(ent.mean())
        pr = 1.0 / Af.pow(2).sum(-1)                                          # [H]
        self.pr += float(pr.mean())

        for k in TOP_KS:
            kk = min(k, n_k)
            self.top_mass[k] += float(torch.topk(Af, kk, dim=-1).values.sum(-1).mean())

        # expected unique-read fraction at each sample budget (collision curve)
        log_keep = torch.log1p(-Af.clamp(max=1 - 1e-12))                      # [H, n_k]
        for s in SAMPLE_BUDGETS:
            uniq = (1.0 - torch.exp(s * log_keep)).sum(-1)                    # [H]
            self.uniq_frac[s] += float((uniq / n_k).mean())

        counts, _ = np.histogram(pr.cpu().numpy(), bins=_PR_BIN_EDGES)
        self.pr_hist = [a + int(b) for a, b in zip(self.pr_hist, counts)]

    def summary(self) -> dict:
        n = max(self.n, 1)
        return {
            "records": self.n,
            "mean_entropy_nats": self.entropy / n,
            "mean_participation_ratio": self.pr / n,
            "mean_n_k": self.nk / n,
            "mean_top_mass": {k: v / n for k, v in self.top_mass.items()},
            "expected_read_fraction": {s: v / n for s, v in self.uniq_frac.items()},
            "pr_hist_bins": _PR_BIN_EDGES,
            "pr_hist_counts": self.pr_hist,
        }


def install_diagnostic(model, stats: ConcentrationStats) -> None:
    def op(q, full_k, full_v, *, scale, layer_idx):
        qd = q[0, :, 0, :]
        K = full_k[0].permute(1, 0, 2).contiguous()
        V = full_v[0].permute(1, 0, 2).contiguous()
        A = attn_weights(qd, K)
        stats.record(A)
        return weighted_sum(A, gqa_expand(V, qd.shape[0])).unsqueeze(0).unsqueeze(2)

    for m in model.modules():
        if isinstance(m, Attention):
            m.decode_attn_op = op


def measure_concentration(model, chunks, *, prefill_len: int) -> dict:
    """Run decode over ``chunks`` recording attention concentration; restore model."""
    from .perplexity import decode_ppl
    from ..models.patch import uninstall

    stats = ConcentrationStats()
    install_diagnostic(model, stats)
    try:
        decode_ppl(model, chunks, prefill_len=prefill_len)  # drives decode; op records A
    finally:
        uninstall(model)
    return stats.summary()


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from .ppl_sweep import _load_model, _wikitext_chunks
    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--max-chunks", type=int, default=4)
    p.add_argument("--chunk-len", type=int, default=2048)
    p.add_argument("--prefill", type=int, default=1024)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = _load_model(args.model, args.device, dtype)
    chunks = _wikitext_chunks(args.model, max_chunks=args.max_chunks,
                              chunk_len=args.chunk_len, device=args.device)
    summary = measure_concentration(model, chunks, prefill_len=args.prefill)
    payload = stamp({"phase": "C-diagnostic", "model": args.model,
                     "chunk_len": args.chunk_len, "prefill_len": args.prefill, **summary})

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"concentration_{payload.get('git_sha', 'nogit')[:8]}.json"
    path.write_text(json.dumps(payload, indent=2))

    print(f"effective tokens (participation ratio): {summary['mean_participation_ratio']:.1f}"
          f"  of n_k≈{summary['mean_n_k']:.0f}")
    print(f"attention entropy: {summary['mean_entropy_nats']:.2f} nats")
    print("top-k cumulative mass: " +
          ", ".join(f"k={k}:{m:.2f}" for k, m in summary["mean_top_mass"].items()))
    print("expected read fraction (collisions): " +
          ", ".join(f"S={s}:{f*100:.1f}%" for s, f in summary["expected_read_fraction"].items()))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
