#!/usr/bin/env bash
# Multi-draw bias-isolation: does the debias LoRA reduce the *systematic* Jensen-gap
# bias (M-draw-averaged), as opposed to the variance-floored single-draw metric?
# For each S, compare plain vs the matching debias LoRA merged in. Idempotent.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

R="src/ssa/results"
M="Qwen/Qwen3-4B"
CFG="--max-chunks 8 --chunk-len 288 --prefill-len 32 --draws 32"

run_if_missing () { local out="$1"; shift
  if [ -f "$out" ]; then echo "[bias] skip (exists): $out"; return 0; fi
  echo "[bias] $(date -Is) RUN → $out"
  "$@" || echo "[bias] FAILED → $out" >&2
}

# Plain code baseline across all four S in one model load.
run_if_missing "$R/bias_code_plain.json" \
  uv run python -m ssa.harness.bias_isolate --model "$M" --corpus code $CFG \
    --S 8 16 32 64 --out "$R/bias_code_plain.json"

# Debiased: one run per S (each LoRA is S-specific).
for S in 8 16 32 64; do
  LORA="$R/debias_lora_code_s${S}.pt"
  [ -f "$LORA" ] || { echo "[bias] no LoRA $LORA, skipping"; continue; }
  run_if_missing "$R/bias_code_debiased_s${S}.json" \
    uv run python -m ssa.harness.bias_isolate --model "$M" --corpus code $CFG \
      --S "$S" --lora "$LORA" --out "$R/bias_code_debiased_s${S}.json"
done

# Broad-workload contrast at S=32.
run_if_missing "$R/bias_wiki_plain.json" \
  uv run python -m ssa.harness.bias_isolate --model "$M" --corpus wikitext $CFG \
    --S 32 --out "$R/bias_wiki_plain.json"
run_if_missing "$R/bias_wiki_debiased_s32.json" \
  uv run python -m ssa.harness.bias_isolate --model "$M" --corpus wikitext $CFG \
    --S 32 --lora "$R/debias_lora_wiki_s32.pt" --out "$R/bias_wiki_debiased_s32.json"

echo "[bias] $(date -Is) bias-isolation chain complete"
