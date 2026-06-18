"""Phase C sweep: perplexity vs read budget across deterministic/stochastic splits.

Headline artifact: PPL increase vs value-read fraction for `dense`, `santa_sys`
at several budgets, and `santa_hybrid` at several (k_h, S_tail) splits at matched
total budget. The end-to-end analog of the Phase A variance-vs-budget result.

Run (user-launched; downloads ~8 GB, GPU, sequential decode is slow):
    uv run python -m ssa.harness.ppl_sweep --model Qwen/Qwen3-4B --max-chunks 16

The actual Qwen3-4B run is intentionally left to the user; ``run_sweep`` is the
testable core and is exercised on a tiny CPU model in the test suite.
"""

from __future__ import annotations

import argparse

import torch

from ..models.patch import install, uninstall
from .perplexity import decode_ppl
from .stamp import stamp

# (impl, cfg). santa_hybrid takes `S` as the tail budget; total budget = k_h + S.
DEFAULT_CONDITIONS = [
    ("dense", {}),
    ("santa_sys", {"S": 16}),
    ("santa_sys", {"S": 64}),
    ("santa_sys", {"S": 256}),
    ("santa_hybrid", {"k_h": 8, "S": 56}),    # total 64, matches santa_sys S=64
    ("santa_hybrid", {"k_h": 32, "S": 32}),   # total 64
]


def _total_budget(impl: str, cfg: dict) -> int | None:
    if impl == "dense":
        return None
    return int(cfg.get("k_h", 0)) + int(cfg.get("S", 0))


def run_sweep(
    model,
    chunks: list[torch.Tensor],
    *,
    conditions=DEFAULT_CONDITIONS,
    prefill_len: int = 32,
    n_runs: int = 3,
) -> list[dict]:
    """Run each condition; sampling conditions are averaged over ``n_runs`` seeds."""
    results = []
    for impl, cfg in conditions:
        if impl == "dense":
            uninstall(model)
            r = decode_ppl(model, chunks, prefill_len=prefill_len)
            results.append({
                "impl": impl, "cfg": cfg, "total_budget": None,
                "ppl_mean": r["ppl"], "ppl_std": 0.0, "read_fraction": 1.0,
                "token_count": r["token_count"], "n_runs": 1,
            })
            continue

        ppls, frac = [], 1.0
        for run in range(n_runs):
            stats = install(model, impl, base_seed=run, **cfg)
            ppls.append(decode_ppl(model, chunks, prefill_len=prefill_len)["ppl"])
            frac = stats.read_fraction
            uninstall(model)
        t = torch.tensor(ppls)
        results.append({
            "impl": impl, "cfg": cfg, "total_budget": _total_budget(impl, cfg),
            "ppl_mean": float(t.mean()), "ppl_std": float(t.std(unbiased=False)),
            "read_fraction": frac, "n_runs": n_runs,
        })
    return results


def _format_table(results: list[dict]) -> str:
    dense = next((r for r in results if r["impl"] == "dense"), None)
    base = dense["ppl_mean"] if dense else None
    lines = [f"{'condition':28s} {'budget':>7s} {'read%':>7s} {'ppl':>10s} {'Δppl%':>8s}"]
    for r in results:
        label = r["impl"] + (f" {r['cfg']}" if r["cfg"] else "")
        budget = "" if r["total_budget"] is None else str(r["total_budget"])
        dppl = "" if base is None else f"{100 * (r['ppl_mean'] - base) / base:+.2f}"
        lines.append(
            f"{label:28s} {budget:>7s} {100*r['read_fraction']:>6.1f}% "
            f"{r['ppl_mean']:>10.4f} {dppl:>8s}"
        )
    return "\n".join(lines)


def _load_model(model_id: str, device: str, dtype: torch.dtype):
    from engine.model import Qwen3Model
    from engine.weights import load_weights

    loaded = load_weights(model_id, dtype=dtype, device="cpu")
    model = Qwen3Model.from_loaded(loaded)
    return model.to(device=device, dtype=dtype).eval()


def _wikitext_chunks(model_id: str, *, max_chunks: int, chunk_len: int, device: str):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    chunks = []
    for i in range(0, ids.numel() - chunk_len, chunk_len):
        chunks.append(ids[i:i + chunk_len].unsqueeze(0).to(device))
        if len(chunks) >= max_chunks:
            break
    return chunks


def main() -> None:
    import json
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--max-chunks", type=int, default=16)
    p.add_argument("--chunk-len", type=int, default=512)
    p.add_argument("--prefill", type=int, default=128)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = _load_model(args.model, args.device, dtype)
    chunks = _wikitext_chunks(
        args.model, max_chunks=args.max_chunks, chunk_len=args.chunk_len, device=args.device
    )
    results = run_sweep(model, chunks, prefill_len=args.prefill, n_runs=args.n_runs)

    payload = stamp({
        "phase": "C", "model": args.model, "dataset": "wikitext-103-raw-v1/test",
        "chunk_len": args.chunk_len, "prefill_len": args.prefill,
        "max_chunks": args.max_chunks, "n_runs": args.n_runs, "results": results,
    })
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"phaseC_ppl_{payload.get('git_sha', 'nogit')[:8]}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(_format_table(results))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
