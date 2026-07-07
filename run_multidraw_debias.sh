#!/usr/bin/env bash
# Sequel to run_bias_isolate.sh: if there's a real systematic bias, does training the
# debiaser on the M-DRAW-averaged loss (targets bias, not variance) reduce it further
# than the single-draw LoRA? Retrain at S=16,32 on code, then bias-isolate the result.
# Waits on the bias chain's completion marker (output file) so the GPU is free.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

R="src/ssa/results"
M="Qwen/Qwen3-4B"
BIAS_LOG="run_bias_isolate.log"

run_if_missing () { local out="$1"; shift
  if [ -f "$out" ]; then echo "[md] skip (exists): $out"; return 0; fi
  echo "[md] $(date -Is) RUN → $out"; "$@" || echo "[md] FAILED → $out" >&2
}

echo "[md] $(date -Is) waiting for bias chain to finish ($BIAS_LOG)"
while ! grep -q "bias-isolation chain complete" "$BIAS_LOG" 2>/dev/null; do sleep 60; done
echo "[md] $(date -Is) bias chain done; starting multi-draw retrain"

for S in 16 32; do
  LORA="$R/debias_lora_code_s${S}_md.pt"
  run_if_missing "$LORA" \
    uv run python -m ssa.harness.debias_train --model "$M" --corpus code \
      --S "$S" --train-draws 6 --pos-per-chunk 12 --steps 300 --val-every 50 --out "$LORA"
  [ -f "$LORA" ] || continue
  run_if_missing "$R/bias_code_debiased_s${S}_md.json" \
    uv run python -m ssa.harness.bias_isolate --model "$M" --corpus code \
      --max-chunks 8 --chunk-len 288 --prefill-len 32 --draws 32 \
      --S "$S" --lora "$LORA" --out "$R/bias_code_debiased_s${S}_md.json"
done

echo "[md] $(date -Is) multi-draw debias chain complete"
