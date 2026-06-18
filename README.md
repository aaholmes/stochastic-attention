# Semi-Stochastic Sparse Attention

A research-validation prototype exploring whether you can make LLM text generation cheaper by **randomly sampling** the attention cache instead of reading all of it — without distorting the model's output on average.

> **Status.** Phase A complete — the statistical core is built and its correctness gates pass. `PROTOTYPE_DESIGN.md` is the authoritative spec. Target hardware: a single 16 GB RTX 5060 Ti (Blackwell).

## The problem, briefly

When an LLM generates text one token at a time, each new token must "attend" to every previous token: the GPU re-reads the entire **KV cache** (a stored key and value vector for every past position, every attention head) from main memory. On a 16 GB consumer GPU this memory traffic — not raw compute — is what caps context length and keeps generation slow.

**Attention is a weighted average of the cached value vectors**, where the weights `A` are a softmax (they sum to 1, so `A` is a probability distribution). The expensive part is touching all of those values every step.

## The idea

If attention is an average over a probability distribution, you can **estimate** it by sampling: draw a handful of value rows according to their attention weights and average them. This is a standard Monte-Carlo estimator, and it is **unbiased** — its expected value equals the true dense attention output — with error that shrinks as roughly `1/S` for `S` samples. We reproduce this published result (the "SANTA" paper) in plain PyTorch, then test three of our own extensions:

1. **Hybrid (deterministic head + sampled tail) — main contribution.** Compute the few highest-weight tokens *exactly* (zero error) and only sample the diffuse remainder. At the same read budget this should cut the estimator's variance — and the win grows the more attention mass sits in those top tokens.
2. **Reuse across steps.** Recycle values sampled on the previous decode step, reweighted by an importance ratio, to avoid re-reading memory — *if* you can prove the reweighting keeps the estimator unbiased and *if* those values are still cache-resident.
3. **Skip-resident.** Sample fresh each step but skip the memory read for blocks already sitting in cache; statistically identical to plain sampling, cheaper in bytes.

## What counts as success

This is about **correctness and statistics, not speed** (the real speedup lives on datacenter hardware we don't have). Concretely, in priority order:

1. **Variance harness** — prove each estimator is unbiased and that its error falls as `1/S`. Runs on tiny random tensors; this is the scientific core. *If the `1/S` slope doesn't reproduce, nothing downstream is valid.*
2. **Accuracy harness** — swap the sampled attention into a real small model (Qwen3-4B, BF16) on long-context tasks and measure task accuracy vs sample budget.
3. **Microbenchmark** — directional kernel-timing only, interpreted through an Amdahl's-law ceiling; we do **not** chase end-to-end speedup numbers on this card. We measure *bytes avoided*, not wall-clock.

## Two design choices worth knowing

- **GPU-friendly block sampling.** Memory is read in contiguous chunks, so we sample ~1–2 KB **blocks** of values rather than scattered single rows. This makes reads coalesced and matches the real unit of on-chip cache residency — but it changes the estimator, so its unbiasedness is derived and tested separately. Variance-vs-block-size at a fixed read budget is a key result.
- **We reuse a sibling project for the model.** The accuracy/microbench phases plug into [`../llms`](../llms) — a from-scratch, HuggingFace-bit-exact Qwen3 inference engine (same author, same GPU) — by swapping its attention op directly, rather than patching HuggingFace internals. The statistics core (`ssa/`) stays standalone.

## What's built so far (Phase A — the statistical core)

The `ssa/` package implements and validates the foundation, on tiny random tensors, with no model in the loop:

- **One swappable interface** `attn(q, K, V, impl=...)` with five implementations: `dense` (exact ground truth), `topk` (biased baseline), and the three unbiased samplers `santa` (i.i.d.), `santa_strat` (stratified), `santa_sys` (systematic).
- **The sampling mechanics** — per-head CDF construction and inverse-CDF index draws (one shared offset per head for systematic), with with-replacement unique-key counting.
- **A measurement harness** that estimates each sampler's Monte-Carlo mean and its variance-vs-budget slope, accumulating in float64 so low-precision roundoff is never mistaken for bias. Results are written to `src/ssa/results/` stamped with git SHA, GPU, and library versions.

**Both correctness gates pass** (24 tests):

1. *Unbiased* — every `santa*` mean matches `dense` to within Monte-Carlo error.
2. *Variance falls as ~1/S* — measured log-log slopes of −1.0 (i.i.d.), −1.3 (stratified), −1.5 (systematic), reproducing the paper's pattern that structured sampling beats i.i.d.

Run it: `uv run pytest` (full suite) and `uv run python -m ssa.harness.variance` (prints the slopes, writes a stamped result file).

**Not yet built:** the hybrid estimator (Idea 1), block sampling, reuse/resident caching (Ideas 2–3), and the real-model accuracy/microbench phases.

## Repository

- `PROTOTYPE_DESIGN.md` — full spec, phase ordering, and correctness gates. **Read this first** (especially §0.5, the current decisions).
- `CLAUDE.md` — orientation for AI coding assistants working in this repo.
- `santa-wiki.html` — background notes on the SANTA method.
