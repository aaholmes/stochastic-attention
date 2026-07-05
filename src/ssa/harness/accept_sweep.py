"""Acceptance / TVD of stochastic-attention decode vs the dense model.

The ppl_sweep measures *perplexity* under sampled attention. This measures the
quantity a speculative-decode verifier (or a bias-correcting LoRA) actually
cares about: how close the sampled-attention next-token distribution is to the
*dense* model's, token for token, as a function of read budget.

For each decode step (teacher-forced, dense prefill then true tokens one at a
time through the sparse decode seam) we compare the sampled-attention logits to
the dense logits of the *same* model at the *same* step:

    acceptance = Σ_i min(p_dense_i, p_stoch_i) = 1 − TVD(p_dense, p_stoch)

This is the single-token spec-decode accept probability (Leviathan et al.) with
the dense model as target and the stochastic model as draft — i.e. exactly the
bias a debiasing LoRA would try to remove. Sampling is unbiased in *attention
output* but biased in *logits* (softmax + MLP are nonlinear), so acceptance < 1
even in expectation; this sweep quantifies that gap vs read-fraction.

Private (ssa): the stochastic-attention debiasing bet. Imports the dense engine
from the sibling public `specd`/`engine`; the novel application stays here.

Run (GPU, sequential decode — slow):
    uv run python -m ssa.harness.accept_sweep --model Qwen/Qwen3-4B \
        --corpus code --max-chunks 8 --chunk-len 288 --prefill-len 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ..models.patch import install, uninstall
from .stamp import stamp

# santa_sys read-budget sweep (S = value rows sampled per head per step).
DEFAULT_S = [4, 8, 16, 32, 64, 128, 256]


@torch.inference_mode()
def _decode_logits(model, ids: torch.Tensor, *, prefill_len: int) -> torch.Tensor:
    """Dense prefill, then feed true tokens one at a time through the decode seam
    (sparse iff an ssa op is installed). Returns [n_steps, vocab] next-token
    logits for the scored steps ``[prefill_len, T-1)``."""
    T = ids.shape[1]
    cache = model.alloc_cache(T + 4)
    model(ids[:, :prefill_len], cache, start_pos=0)  # dense prefill (T>1 branch)
    out = []
    for t in range(prefill_len, T - 1):
        logits = model(ids[:, t:t + 1], cache)       # decode step (sparse if installed)
        out.append(logits[0, -1, :])
    return torch.stack(out)                            # [steps, vocab]


def _compare(dense: torch.Tensor, stoch: torch.Tensor, true_next: torch.Tensor) -> dict:
    """Per-step acceptance / TVD / top-1 agreement (stoch vs dense) + student NLL."""
    p_d = dense.float().softmax(-1)
    p_s = stoch.float().softmax(-1)
    accept = torch.minimum(p_d, p_s).sum(-1)               # [steps]
    top1 = (dense.argmax(-1) == stoch.argmax(-1)).float()  # [steps]
    nll = F.cross_entropy(stoch.float(), true_next, reduction="none")  # student PPL
    return {
        "accept_sum": float(accept.sum()), "top1_sum": float(top1.sum()),
        "nll_sum": float(nll.sum()), "n": int(accept.numel()),
    }


def _load_chunks(model_id: str, corpus: str, *, max_chunks: int, chunk_len: int, device: str):
    if corpus == "code":
        from mla.calibrate import load_code_chunks
        chunks = load_code_chunks(n_samples=max_chunks, chunk_tokens=chunk_len,
                                  tokenizer_id=model_id, skip=20000)
    else:
        from mla.calibrate import load_wikitext103_chunks
        chunks = load_wikitext103_chunks(n_samples=max_chunks, chunk_tokens=chunk_len,
                                         tokenizer_id=model_id, split="validation")
    return [c.to(device) for c in chunks]


def run(model, chunks, *, s_values, prefill_len: int, base_seed: int = 0) -> list[dict]:
    """Dense reference once, then santa_sys at each S; acceptance/TVD/top1/PPL vs dense."""
    # 1) Dense teacher logits per chunk (store on CPU to free GPU for the S loop).
    uninstall(model)
    dense_logits = [_decode_logits(model, ids, prefill_len=prefill_len).cpu() for ids in chunks]
    true_next = [ids[0, prefill_len + 1:].cpu() for ids in chunks]

    results = []
    for S in s_values:
        stats = install(model, "santa_sys", base_seed=base_seed, S=S)
        agg = {"accept_sum": 0.0, "top1_sum": 0.0, "nll_sum": 0.0, "n": 0}
        for ids, d_cpu, tn in zip(chunks, dense_logits, true_next):
            s_logits = _decode_logits(model, ids, prefill_len=prefill_len)
            c = _compare(d_cpu.to(s_logits.device), s_logits, tn.to(s_logits.device))
            for k in agg:
                agg[k] += c[k]
        uninstall(model)
        n = max(agg["n"], 1)
        acc = agg["accept_sum"] / n
        results.append({
            "S": S, "read_fraction": stats.read_fraction,
            "acceptance": acc, "tvd": 1.0 - acc,
            "top1_agree": agg["top1_sum"] / n,
            "ppl": float(torch.tensor(agg["nll_sum"] / n).exp()),
            "n_steps": agg["n"],
        })
        r = results[-1]
        print(f"[accept] S={S:<4d} reads={100*r['read_fraction']:5.1f}%  "
              f"accept={100*acc:5.1f}%  tvd={r['tvd']:.4f}  top1={100*r['top1_agree']:5.1f}%  "
              f"ppl={r['ppl']:.3f}", flush=True)
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--corpus", default="code", choices=["code", "wikitext"])
    p.add_argument("--max-chunks", type=int, default=8)
    p.add_argument("--chunk-len", type=int, default=288)
    p.add_argument("--prefill-len", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lora", default=None, help="trained debias-LoRA checkpoint to load before sweeping")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--out", default="src/ssa/results/accept_sweep.json")
    args = p.parse_args()

    from engine.model import Qwen3Model
    from engine.weights import load_weights

    print(f"[accept] loading {args.model}", flush=True)
    loaded = load_weights(args.model, dtype=torch.bfloat16, device="cpu")
    model = Qwen3Model.from_loaded(loaded).to(dtype=torch.bfloat16, device=args.device).eval()
    del loaded
    torch.cuda.empty_cache()

    if args.lora:
        from mla.heal import load_trainable, merge_lora, wrap_lora
        wrap_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        load_trainable(model, args.lora)
        merge_lora(model)  # fold in → plain model, same decode path
        print(f"[accept] loaded + merged debias-LoRA {args.lora}", flush=True)

    chunks = _load_chunks(args.model, args.corpus, max_chunks=args.max_chunks,
                          chunk_len=args.chunk_len, device=args.device)
    print(f"[accept] {len(chunks)} {args.corpus} chunks × {args.chunk_len} tok "
          f"(prefill {args.prefill_len})", flush=True)

    results = run(model, chunks, s_values=DEFAULT_S, prefill_len=args.prefill_len)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = stamp({"model": args.model, "corpus": args.corpus,
                     "chunk_len": args.chunk_len, "prefill_len": args.prefill_len,
                     "results": results})
    out.write_text(json.dumps(payload, indent=2))
    print(f"[accept] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
