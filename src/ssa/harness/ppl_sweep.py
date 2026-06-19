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
# Frontier set: santa_sys swept across budgets, plus santa_hybrid at two matched
# total budgets (64 and 256) with several head/tail splits — to test whether the
# hybrid overtakes plain systematic at long context.
DEFAULT_CONDITIONS = [
    ("dense", {}),
    ("santa_sys", {"S": 8}),
    ("santa_sys", {"S": 16}),
    ("santa_sys", {"S": 32}),
    ("santa_sys", {"S": 64}),
    ("santa_sys", {"S": 128}),
    ("santa_sys", {"S": 256}),
    ("santa_sys", {"S": 512}),
    ("santa_hybrid", {"k_h": 8, "S": 56}),     # total 64
    ("santa_hybrid", {"k_h": 16, "S": 48}),    # total 64
    ("santa_hybrid", {"k_h": 32, "S": 32}),    # total 64
    ("santa_hybrid", {"k_h": 32, "S": 224}),   # total 256
    ("santa_hybrid", {"k_h": 64, "S": 192}),   # total 256
    ("santa_hybrid", {"k_h": 128, "S": 128}),  # total 256
]

# Cheap-end set: probes the low-budget regime where the deterministic head should
# pay off, plus a det:stoch ratio sweep and the biased top-k baseline. Deliberately
# omits santa_sys (reuse the existing de-risk frontier — overlay at plot time).
CHEAP_CONDITIONS = [
    ("dense", {}),
    # top-k biased baseline (deterministic; reads exactly k rows)
    ("topk", {"k": 1}), ("topk", {"k": 2}), ("topk", {"k": 4}), ("topk", {"k": 8}),
    ("topk", {"k": 16}), ("topk", {"k": 32}), ("topk", {"k": 64}),
    # semistoch "1 exact + rest stochastic" across budgets — the cheap-end frontier
    ("santa_hybrid", {"k_h": 1, "S": 3}),     # total 4
    ("santa_hybrid", {"k_h": 1, "S": 7}),     # total 8
    ("santa_hybrid", {"k_h": 1, "S": 15}),    # total 16
    ("santa_hybrid", {"k_h": 1, "S": 31}),    # total 32
    ("santa_hybrid", {"k_h": 1, "S": 63}),    # total 64
    # det:stoch ratio sweep at fixed total budgets 16 and 32
    ("santa_hybrid", {"k_h": 4, "S": 12}),    # total 16
    ("santa_hybrid", {"k_h": 8, "S": 8}),     # total 16
    ("santa_hybrid", {"k_h": 8, "S": 24}),    # total 32
    ("santa_hybrid", {"k_h": 16, "S": 16}),   # total 32
]

CONDITION_PRESETS = {"full": DEFAULT_CONDITIONS, "cheap": CHEAP_CONDITIONS}


def _total_budget(impl: str, cfg: dict) -> int | None:
    if impl == "dense":
        return None
    return int(cfg.get("k_h", 0)) + int(cfg.get("S", 0))


def _run_condition(model, chunks, impl, cfg, *, prefill_len, n_runs) -> dict:
    """Compute one condition's PPL (+ timing, read fraction). Dense uses 1 run."""
    import time
    t0 = time.time()
    if impl == "dense":
        uninstall(model)
        r = decode_ppl(model, chunks, prefill_len=prefill_len)
        dt = time.time() - t0
        return {
            "impl": impl, "cfg": cfg, "total_budget": None,
            "ppl_mean": r["ppl"], "ppl_std": 0.0, "read_fraction": 1.0,
            "token_count": r["token_count"], "n_runs": 1,
            "wall_seconds": dt, "sec_per_token": dt / max(r["token_count"], 1),
        }
    if impl == "topk":  # deterministic biased baseline — single run, reads top-k rows
        stats = install(model, "topk", **cfg)
        r = decode_ppl(model, chunks, prefill_len=prefill_len)
        uninstall(model)
        dt = time.time() - t0
        return {
            "impl": impl, "cfg": cfg, "total_budget": int(cfg.get("k", 0)),
            "ppl_mean": r["ppl"], "ppl_std": 0.0, "read_fraction": stats.read_fraction,
            "token_count": r["token_count"], "n_runs": 1,
            "wall_seconds": dt, "sec_per_token": dt / max(r["token_count"], 1),
        }
    ppls, frac, tokens = [], 1.0, 0
    for run in range(n_runs):
        stats = install(model, impl, base_seed=run, **cfg)
        r = decode_ppl(model, chunks, prefill_len=prefill_len)
        ppls.append(r["ppl"])
        tokens += r["token_count"]
        frac = stats.read_fraction
        uninstall(model)
    dt = time.time() - t0
    t = torch.tensor(ppls)
    return {
        "impl": impl, "cfg": cfg, "total_budget": _total_budget(impl, cfg),
        "ppl_mean": float(t.mean()), "ppl_std": float(t.std(unbiased=False)),
        "read_fraction": frac, "n_runs": n_runs,
        "wall_seconds": dt, "sec_per_token": dt / max(tokens, 1),
    }


def run_sweep(
    model,
    chunks: list[torch.Tensor],
    *,
    conditions=DEFAULT_CONDITIONS,
    prefill_len: int = 32,
    n_runs: int = 3,
    on_condition=None,
) -> list[dict]:
    """Run each condition; sampling conditions are averaged over ``n_runs`` seeds.

    Each result carries ``wall_seconds``/``sec_per_token``. A failing condition
    is recorded as an ``error`` entry and the sweep continues (so one OOM doesn't
    sink an overnight run). ``on_condition(result, results_so_far)`` is called
    after every condition — used for crash-safe checkpointing + live progress.
    """
    results: list[dict] = []
    for impl, cfg in conditions:
        try:
            result = _run_condition(model, chunks, impl, cfg, prefill_len=prefill_len, n_runs=n_runs)
        except Exception as exc:  # keep going; record what failed
            uninstall(model)
            result = {"impl": impl, "cfg": cfg, "total_budget": _total_budget(impl, cfg),
                      "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        if on_condition is not None:
            on_condition(result, results)
    return results


def _row(r: dict, base: float | None) -> str:
    label = r["impl"] + (f" {r['cfg']}" if r["cfg"] else "")
    if "error" in r:
        return f"{label:28s} {'ERROR: ' + r['error']}"
    budget = "" if r["total_budget"] is None else str(r["total_budget"])
    dppl = "" if base is None else f"{100 * (r['ppl_mean'] - base) / base:+.2f}"
    return (
        f"{label:28s} {budget:>7s} {100*r['read_fraction']:>6.1f}% "
        f"{r['ppl_mean']:>10.4f} {dppl:>8s} "
        f"{r.get('wall_seconds', 0):>7.1f} {1000*r.get('sec_per_token', 0):>7.1f}"
    )


def _header() -> str:
    return (f"{'condition':28s} {'budget':>7s} {'read%':>7s} {'ppl':>10s} {'Δppl%':>8s} "
            f"{'sec':>7s} {'ms/tok':>7s}")


def _format_table(results: list[dict]) -> str:
    dense = next((r for r in results if r["impl"] == "dense" and "error" not in r), None)
    base = dense["ppl_mean"] if dense else None
    lines = [_header()] + [_row(r, base) for r in results]
    total = sum(r.get("wall_seconds", 0) for r in results)
    lines.append(f"{'TOTAL':28s} {'':>7s} {'':>7s} {'':>10s} {'':>8s} {total:>7.1f}")
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
    ds = load_dataset("wikitext", "wikitext-103-v1", split="test")
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
    p.add_argument("--preset", default="full", choices=list(CONDITION_PRESETS),
                   help="condition set: 'full' frontier or 'cheap' low-budget probe")
    p.add_argument("--tag", default="", help="suffix for the output filename")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = _load_model(args.model, args.device, dtype)
    chunks = _wikitext_chunks(
        args.model, max_chunks=args.max_chunks, chunk_len=args.chunk_len, device=args.device
    )

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    def make_payload(results, complete):
        return stamp({
            "phase": "C", "model": args.model, "dataset": "wikitext-103-v1/test",
            "chunk_len": args.chunk_len, "prefill_len": args.prefill,
            "max_chunks": args.max_chunks, "n_runs": args.n_runs,
            "complete": complete, "results": results,
        })

    sha = make_payload([], False).get("git_sha", "nogit")[:8]
    tag = f"_{args.tag}" if args.tag else ""
    path = out_dir / f"phaseC_ppl_{sha}{tag}.json"

    base = {"v": None}
    print(_header(), flush=True)

    def on_condition(result, results_so_far):
        if result["impl"] == "dense" and "error" not in result:
            base["v"] = result["ppl_mean"]
        print(_row(result, base["v"]), flush=True)
        # crash-safe: persist after every condition
        path.write_text(json.dumps(make_payload(results_so_far, False), indent=2))

    results = run_sweep(
        model, chunks, conditions=CONDITION_PRESETS[args.preset],
        prefill_len=args.prefill, n_runs=args.n_runs, on_condition=on_condition
    )
    path.write_text(json.dumps(make_payload(results, True), indent=2))
    print("\n" + _format_table(results))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
