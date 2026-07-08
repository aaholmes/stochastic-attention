"""Isolate the *systematic* logit bias of stochastic attention from per-draw variance.

Motivation. Sampled attention is unbiased in attention *output* but biased in
*logits*: logits are a nonlinear function of the attention output, so
`E[softmax(logits)] ≠ softmax(dense logits)` (a Jensen gap). That systematic bias
is the correctable part — the thing a debiasing LoRA can, in principle, remove.

The trap the earlier `accept_sweep` fell into. A SINGLE stochastic draw per step
measures `TVD(p_dense, p_1draw)`, which mixes bias and variance. With santa_sys the
attention-output error is `ε`, `Var(ε) ~ 1/S`, so:

    single-draw fluctuation  ~ O(1/√S)   (mean-zero, variance)
    systematic bias          ~ O(1/S)    (the Jensen gap, correctable)

Since `1/√S ≫ 1/S`, single-draw TVD is dominated by variance and *masks* the bias.
A deterministic LoRA can only move the mean (remove bias), so it barely shifts a
variance-floored metric — which is why the single-draw debias result looked null.

This harness isolates the bias. For each decode step it averages `M` independent
stochastic draws to estimate `p̄ = Ê[softmax(stoch logits)]`, then reports:

    bias      = TVD(p̄, p_dense)                 ← systematic Jensen gap (variance averaged out)
    per_draw  = mean_m TVD(softmax(stoch_m), p_dense)   ← bias + variance (what accept_sweep saw)
    variance  = per_draw − bias                 ← the part a LoRA can never remove

Run it twice — once plain, once with `--lora <debias ckpt>` merged — and the real
test of the debiaser is whether it lowers `bias`, not `per_draw`.

Note: `p̄` from finite `M` still carries `O(1/(M·S))` residual variance, so `bias`
is a slight over-estimate; raise `--draws` to tighten it. The plain-vs-LoRA
*difference* in `bias` is the signal, and that residual affects both equally.

Run (GPU, sequential decode — slow; M× the cost of accept_sweep):
    uv run python -m ssa.harness.bias_isolate --model Qwen/Qwen3-4B --corpus code \
        --S 32 --draws 32 --out src/ssa/results/bias_code_s32.json
    uv run python -m ssa.harness.bias_isolate --model Qwen/Qwen3-4B --corpus code \
        --S 32 --draws 32 --lora src/ssa/results/debias_lora_code_s32.pt \
        --out src/ssa/results/bias_code_s32_debiased.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..models.patch import install, uninstall
from .accept_sweep import _decode_logits, _load_chunks
from .stamp import stamp


def _tvd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-step total-variation distance between two prob tensors [steps, vocab]."""
    return 0.5 * (p - q).abs().sum(-1)


def run(model, chunks, *, s_values, draws: int, prefill_len: int, base_seed: int = 0) -> list[dict]:
    """For each S: dense reference, then M-draw averaged stochastic distribution,
    reporting the bias / per-draw / variance split (means over all scored steps)."""
    # Dense reference distributions (LoRA-on if a debias LoRA was merged in).
    uninstall(model)
    dense_p = [_decode_logits(model, ids, prefill_len=prefill_len).float().softmax(-1).cpu()
               for ids in chunks]

    results = []
    for S in s_values:
        bias_sum, perdraw_sum, n = 0.0, 0.0, 0
        read_fraction = 1.0
        for ids, d_cpu in zip(chunks, dense_p):
            d = d_cpu.to(ids.device)                       # [steps, vocab] dense probs
            p_bar = torch.zeros_like(d)                    # running mean of softmaxes
            perdraw = torch.zeros(d.shape[0], device=d.device)
            for m in range(draws):
                stats = install(model, "santa_sys", base_seed=base_seed + 1000 * m + S, S=S)
                p_m = _decode_logits(model, ids, prefill_len=prefill_len).float().softmax(-1)
                read_fraction = stats.read_fraction  # read AFTER decode populates the counter
                uninstall(model)
                p_bar += p_m
                perdraw += _tvd(p_m, d)
            p_bar /= draws
            perdraw /= draws
            bias_sum += float(_tvd(p_bar, d).sum())
            perdraw_sum += float(perdraw.sum())
            n += d.shape[0]

        n = max(n, 1)
        bias = bias_sum / n
        per_draw = perdraw_sum / n
        results.append({
            "S": S, "read_fraction": read_fraction, "draws": draws,
            "bias": bias, "per_draw_tvd": per_draw, "variance": per_draw - bias,
            "bias_frac_of_per_draw": bias / per_draw if per_draw else 0.0,
            "n_steps": n,
        })
        r = results[-1]
        print(f"[bias] S={S:<4d} reads={100*r['read_fraction']:5.1f}%  "
              f"bias={r['bias']:.4f}  per_draw={r['per_draw_tvd']:.4f}  "
              f"variance={r['variance']:.4f}  bias/per_draw={100*r['bias_frac_of_per_draw']:4.1f}%",
              flush=True)
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--corpus", default="code", choices=["code", "wikitext"])
    p.add_argument("--S", type=int, nargs="+", default=[32],
                   help="read budget(s) to profile (match the debias LoRA's S)")
    p.add_argument("--draws", type=int, default=32, help="stochastic draws averaged per step")
    p.add_argument("--max-chunks", type=int, default=8)
    p.add_argument("--chunk-len", type=int, default=288)
    p.add_argument("--prefill-len", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lora", default=None, help="debias-LoRA checkpoint to merge before profiling")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--out", default="src/ssa/results/bias_isolate.json")
    args = p.parse_args()

    from engine.model import Qwen3Model
    from engine.weights import load_weights

    print(f"[bias] loading {args.model}", flush=True)
    loaded = load_weights(args.model, dtype=torch.bfloat16, device="cpu")
    model = Qwen3Model.from_loaded(loaded).to(dtype=torch.bfloat16, device=args.device).eval()
    del loaded
    torch.cuda.empty_cache()

    if args.lora:
        from mla.heal import load_trainable, merge_lora, wrap_lora
        wrap_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        load_trainable(model, args.lora)
        merge_lora(model)  # fold in → plain model, same decode path
        print(f"[bias] loaded + merged debias-LoRA {args.lora}", flush=True)

    chunks = _load_chunks(args.model, args.corpus, max_chunks=args.max_chunks,
                          chunk_len=args.chunk_len, device=args.device)
    print(f"[bias] {len(chunks)} {args.corpus} chunks × {args.chunk_len} tok "
          f"(prefill {args.prefill_len}), {args.draws} draws/step", flush=True)

    results = run(model, chunks, s_values=args.S, draws=args.draws, prefill_len=args.prefill_len)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = stamp({"model": args.model, "corpus": args.corpus, "draws": args.draws,
                     "lora": args.lora, "chunk_len": args.chunk_len,
                     "prefill_len": args.prefill_len, "results": results})
    out.write_text(json.dumps(payload, indent=2))
    print(f"[bias] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
