#!/usr/bin/env bash
# Bias isolation: does a debiasing LoRA (low-rank adapter) reduce the systematic
# Jensen-gap bias of sampled attention, measured by averaging M draws per step so the
# per-draw variance cancels? For each S, trains an S-specific adapter with
# ssa.harness.debias_train (weights go to *.pt, not tracked in git), then compares
# plain and adapted models. Each step is skipped if its output already exists.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

R="src/ssa/results"
M="Qwen/Qwen3-4B"
CFG="--max-chunks 8 --chunk-len 288 --prefill-len 32 --draws 32"
TRAIN="--steps 300 --pos-per-chunk 32 --val-every 50"

run_if_missing () { local out="$1"; shift
  if [ -f "$out" ]; then echo "[bias] skip (exists): $out"; return 0; fi
  echo "[bias] $(date -Is) RUN → $out"
  "$@" || echo "[bias] FAILED → $out" >&2
}

# Plain code baseline across all four S in one model load.
run_if_missing "$R/bias_code_plain.json" \
  uv run python -m ssa.harness.bias_isolate --model "$M" --corpus code $CFG \
    --S 8 16 32 64 --out "$R/bias_code_plain.json"

# Debiased: train one adapter per S, then measure its bias.
for S in 8 16 32 64; do
  LORA="$R/debias_lora_code_s${S}.pt"
  run_if_missing "$LORA" \
    uv run python -m ssa.harness.debias_train --model "$M" --corpus code \
      --S "$S" $TRAIN --out "$LORA"
  [ -f "$LORA" ] || continue
  run_if_missing "$R/bias_code_debiased_s${S}.json" \
    uv run python -m ssa.harness.bias_isolate --model "$M" --corpus code $CFG \
      --S "$S" --lora "$LORA" --out "$R/bias_code_debiased_s${S}.json"
done

# Second corpus (WikiText-103) at S=32.
LORA="$R/debias_lora_wiki_s32.pt"
run_if_missing "$R/bias_wiki_plain.json" \
  uv run python -m ssa.harness.bias_isolate --model "$M" --corpus wikitext $CFG \
    --S 32 --out "$R/bias_wiki_plain.json"
run_if_missing "$LORA" \
  uv run python -m ssa.harness.debias_train --model "$M" --corpus wikitext \
    --S 32 $TRAIN --out "$LORA"
[ -f "$LORA" ] && run_if_missing "$R/bias_wiki_debiased_s32.json" \
  uv run python -m ssa.harness.bias_isolate --model "$M" --corpus wikitext $CFG \
    --S 32 --lora "$LORA" --out "$R/bias_wiki_debiased_s32.json"

echo "[bias] $(date -Is) bias-isolation chain complete"
