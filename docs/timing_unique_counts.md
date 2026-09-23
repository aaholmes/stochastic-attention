# Timing: vectorizing `unique_counts`

The decode-time read-fraction metric counts the distinct value rows each sampled
attention step reads (`ssa.sampling.draws.unique_counts`). The original version
looped over heads in Python calling `torch.unique` per head — on the decode hot
path that is ~32 tiny GPU ops **with host syncs, per layer, per token** (36 layers
× 32 heads on Qwen3-4B), which dominated the sampling overhead.

The fix sorts each `[H, S]` row once and counts value changes
(`distinct = 1 + #jumps`) — fully vectorized, no per-head loop, no host sync.

## Calibration (identical config, one machine)

`Qwen/Qwen3-4B`, WikiText-103 test, `--max-chunks 2 --chunk-len 256 --prefill 64
--n-runs 1`, RTX 5060 Ti (BF16). The two runs differ only in the `unique_counts`
implementation.

| condition                 | read% | PPL     | ms/tok **before** | ms/tok **after** | speedup |
|---------------------------|------:|--------:|------------------:|-----------------:|--------:|
| dense (no sampling)       | 100.0 | 16.5578 |              26.6 |             26.6 |   1.00× |
| santa_sys S=16            |   4.1 | 18.0138 |              64.1 |             30.7 |   2.09× |
| santa_sys S=64            |  10.0 | 16.9055 |              64.4 |             30.5 |   2.11× |
| santa_sys S=256           |  21.5 | 16.5916 |              64.9 |             31.3 |   2.07× |
| santa_hybrid k_h=8,S=56   |  29.0 | 16.7056 |              66.6 |             32.3 |   2.06× |
| santa_hybrid k_h=32,S=32  |  38.8 | 16.6148 |              66.6 |             32.1 |   2.08× |
| **total wall (s)**        |       |         |         **134.9** |         **70.1** | **1.92×** |

## Takeaways

- **PPL is unchanged** to four decimals — a pure performance fix, correctness preserved
  (also covered by an equivalence test vs the `torch.unique` reference).
- **Dense is unaffected** (it never calls `unique_counts`), confirming the win is
  isolated to the sampling path.
- Sampling steps went **~65 → ~31 ms/tok (~2.1×)**. The sparse-specific overhead
  over dense fell from **~38 → ~5 ms/tok**, so `unique_counts` was ~**85–90%** of
  the per-step sampling cost.
- Extrapolated to the default full sweep (16×512 tokens, 6 conditions, 3 seeds),
  this roughly **halves** wall time (order ~100 min → ~50 min); the remaining cost
  is the sequential decode + sampling, which is memory-bound on this card.
