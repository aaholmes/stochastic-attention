"""Train a LoRA to DEBIAS stochastic-attention decode toward the dense model.

The novel bet: sampled attention is unbiased in *attention output* but biased in
*logits* (softmax + MLP are nonlinear → a per-draw Jensen gap). That systematic
component is correctable. We attach a LoRA to the model and train it, teacher-
forced through the decode seam, to minimize TVD between the sampled-attention
logits (student, LoRA on) and the original dense logits (teacher, LoRA off) — i.e.
to maximize single-token acceptance of the cheap sampled model against the exact
model. If it works, the usable read-fraction drops at fixed acceptance.

Tractable training. A decode step at position t reads cache[:t]; the KV cache is
identical whether attention is dense or sampled (K/V are input projections, the
attention impl only changes the *output*). So we prefill the whole chunk ONCE
with no-grad, then score positions in DECREASING order: each grad'd stochastic
decode step at t reads only frozen (no-grad) cache entries < t, giving shallow,
independent per-step graphs — no deep cross-step graph, cheap backward.

Private (ssa). Reuses the LoRA + optimizer machinery from the public `llms`
heal harness (`mla.heal`); the stochastic-attention application stays here.

Run (GPU, after the bias-curve sweep):
    uv run python -m ssa.harness.debias_train --model Qwen/Qwen3-4B --corpus code \
        --S 32 --steps 300 --out src/ssa/results/debias_lora_code_s32.pt
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mla.heal import LoraLinear, build_optimizer, cosine_warmup_multiplier, save_trainable, wrap_lora

from ..models.patch import install, uninstall


@contextlib.contextmanager
def lora_disabled(model):
    """Temporarily zero every LoRA scaling → the model behaves as the original."""
    saved = []
    for m in model.modules():
        if isinstance(m, LoraLinear):
            saved.append((m, m.scaling))
            m.scaling = 0.0
    try:
        yield
    finally:
        for m, s in saved:
            m.scaling = s


@torch.no_grad()  # no_grad (not inference_mode): the result is a *constant* in the student graph
def _teacher_logits(model, ids, *, prefill_len):
    """Original dense next-token logits at scored positions [prefill_len, T-1)."""
    with lora_disabled(model):
        cache = model.alloc_cache(ids.shape[1] + 4)
        out = model(ids[:, : ids.shape[1] - 1], cache, start_pos=0)  # dense parallel forward
    return out[0, prefill_len:, :].contiguous()  # [n_pos, vocab]


def _tvd_loss(teacher_logits, student_logits):
    p_t = teacher_logits.float().softmax(-1)
    p_s = student_logits.float().softmax(-1)
    return 1.0 - torch.minimum(p_t, p_s).sum(-1)  # per-position TVD


def _student_step_logits(model, ids, cache, positions, *, S, base_seed, step):
    """Sampled-attention logits at each position (grad on), reading the frozen cache.
    Positions scored in DECREASING order so no step reads another's grad cache write."""
    logits = {}
    for t in sorted(positions, reverse=True):
        gen_seed = (base_seed * 100_003 + step * 17 + t) % (2**31 - 1)
        install(model, "santa_sys", base_seed=gen_seed, S=S)   # fresh sampling RNG per position
        out = model(ids[:, t : t + 1], cache, start_pos=t)     # single sampled decode step (grad)
        logits[t] = out[0, -1, :]
    uninstall(model)
    return logits


def _student_step_probs(model, ids, cache, positions, *, S, base_seed, step, draws):
    """Mean student softmax over `draws` independent stochastic draws per position
    (grad on). Averaging inside the loss targets the *systematic* (expected) bias —
    the Jensen gap — instead of the per-draw variance a single draw is floored by."""
    acc = {t: None for t in positions}
    for d in range(draws):
        for t in sorted(positions, reverse=True):
            gen_seed = (base_seed * 100_003 + step * 17 + d * 7919 + t) % (2**31 - 1)
            install(model, "santa_sys", base_seed=gen_seed, S=S)
            out = model(ids[:, t : t + 1], cache, start_pos=t)
            p = out[0, -1, :].float().softmax(-1)
            acc[t] = p if acc[t] is None else acc[t] + p
    uninstall(model)
    return {t: acc[t] / draws for t in positions}


def _chunks(model_id, corpus, *, n, seq_len, skip):
    if corpus == "code":
        from mla.calibrate import load_code_chunks
        return load_code_chunks(n_samples=n, chunk_tokens=seq_len, tokenizer_id=model_id, skip=skip)
    from mla.calibrate import load_wikitext103_chunks
    return load_wikitext103_chunks(n_samples=n, chunk_tokens=seq_len, tokenizer_id=model_id,
                                   split="train")


@torch.inference_mode()
def _val_acceptance(model, chunks, *, S, prefill_len, base_seed=7):
    """Mean acceptance (1−TVD) of sampled+LoRA decode vs dense, over val chunks."""
    tot, n = 0.0, 0
    for ids in chunks:
        ids = ids.to(next(model.parameters()).device)
        teacher = _teacher_logits(model, ids, prefill_len=prefill_len)
        cache = model.alloc_cache(ids.shape[1] + 4)
        model(ids[:, :prefill_len], cache, start_pos=0)  # LoRA-on prefill (no grad, inference_mode)
        T = ids.shape[1]
        for i, t in enumerate(range(prefill_len, T - 1)):
            install(model, "santa_sys", base_seed=base_seed + t, S=S)
            out = model(ids[:, t : t + 1], cache, start_pos=t)
            uninstall(model)
            acc = torch.minimum(teacher[i].float().softmax(-1),
                                out[0, -1].float().softmax(-1)).sum()
            tot += float(acc); n += 1
    return tot / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--corpus", default="code", choices=["code", "wikitext"])
    p.add_argument("--S", type=int, default=32, help="sampled read budget to debias at")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--train-draws", type=int, default=1,
                   help="stochastic draws averaged per position in the loss (>1 targets bias, not variance)")
    p.add_argument("--seq-len", type=int, default=288)
    p.add_argument("--prefill-len", type=int, default=32)
    p.add_argument("--pos-per-chunk", type=int, default=32, help="scored positions per prefill")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--val-chunks", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="src/ssa/results/debias_lora.pt")
    args = p.parse_args()

    from engine.model import Qwen3Model
    from engine.weights import load_weights

    print(f"[debias] loading {args.model}", flush=True)
    loaded = load_weights(args.model, dtype=torch.bfloat16, device="cpu")
    model = Qwen3Model.from_loaded(loaded).to(dtype=torch.bfloat16, device=args.device)
    del loaded; torch.cuda.empty_cache()

    n_wrapped = wrap_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    for n, prm in model.named_parameters():
        prm.requires_grad_("lora_" in n)
    lora_params = [prm for n, prm in model.named_parameters() if prm.requires_grad]
    print(f"[debias] wrapped {n_wrapped} LoRA layers, {sum(p.numel() for p in lora_params)} params", flush=True)
    optim = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.0)

    train = _chunks(args.model, args.corpus, n=args.steps + 8, seq_len=args.seq_len, skip=0)
    val = _chunks(args.model, args.corpus, n=args.val_chunks, seq_len=args.seq_len, skip=15000)
    dev = args.device

    base = _val_acceptance(model, val, S=args.S, prefill_len=args.prefill_len)
    print(f"[debias] pre-train val acceptance @S={args.S}: {100*base:.2f}%", flush=True)

    best, best_state = base, None
    for step in range(args.steps):
        ids = train[step % len(train)].to(dev)
        T = ids.shape[1]
        teacher = _teacher_logits(model, ids, prefill_len=args.prefill_len)  # [n_pos, vocab], no grad
        cache = model.alloc_cache(T + 4)
        with torch.no_grad():
            # Populate the WHOLE cache (LoRA-on, dense parallel, no grad). K/V are input
            # projections independent of attention impl, so cache[:t] here is identical to
            # what sequential decode would build — each scored step reads a graph-free cache.
            model(ids[:, : T - 1], cache, start_pos=0)
        avail = list(range(args.prefill_len, T - 1))
        k = min(args.pos_per_chunk, len(avail))
        positions = avail[-k:]  # a contiguous decreasing-scoreable tail
        if args.train_draws > 1:
            probs = _student_step_probs(model, ids, cache, positions, S=args.S,
                                        base_seed=1, step=step, draws=args.train_draws)
            losses = []
            for t in positions:
                p_t = teacher[t - args.prefill_len].float().softmax(-1)
                losses.append(1.0 - torch.minimum(p_t, probs[t]).sum(-1))  # TVD(teacher, E[student])
            loss = torch.stack(losses).mean()
        else:
            stu = _student_step_logits(model, ids, cache, positions, S=args.S, base_seed=1, step=step)
            losses = []
            for t in positions:
                i = t - args.prefill_len
                losses.append(_tvd_loss(teacher[i], stu[t]))
            loss = torch.stack(losses).mean()
        (loss).backward()
        for g in optim.param_groups:
            g["lr"] = args.lr * cosine_warmup_multiplier(step, args.steps, args.warmup_frac)
        optim.step(); optim.zero_grad(set_to_none=True)
        if step % 10 == 0:
            print(f"[debias] step {step}/{args.steps} tvd_loss={float(loss):.4f}", flush=True)
        if step > 0 and step % args.val_every == 0:
            acc = _val_acceptance(model, val, S=args.S, prefill_len=args.prefill_len)
            print(f"[debias]   val acceptance @S={args.S}: {100*acc:.2f}% (base {100*base:.2f}%)", flush=True)
            if acc > best:
                best = acc
                best_state = {n: prm.detach().cpu().clone() for n, prm in model.named_parameters() if prm.requires_grad}

    final = _val_acceptance(model, val, S=args.S, prefill_len=args.prefill_len)
    print(f"[debias] final val acceptance @S={args.S}: {100*final:.2f}% (base {100*base:.2f}%, "
          f"best {100*best:.2f}%)", flush=True)

    # restore the best checkpoint before saving (final may have regressed)
    if best_state is not None:
        with torch.no_grad():
            for n, prm in model.named_parameters():
                if n in best_state:
                    prm.copy_(best_state[n].to(prm.device, prm.dtype))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    save_trainable(model, out)
    (out.with_suffix(".json")).write_text(json.dumps({
        "model": args.model, "corpus": args.corpus, "S": args.S, "steps": args.steps,
        "base_acceptance": base, "final_acceptance": final, "best_acceptance": best,
    }, indent=2))
    print(f"[debias] saved LoRA → {out}", flush=True)


if __name__ == "__main__":
    main()
