"""Adapter + installer: plug ssa estimators into the engine's decode seam.

The engine calls ``decode_attn_op(q, full_k, full_v, *, scale, layer_idx)`` for
single-query decode steps with:
  - ``q``        : [1, H, 1, d]
  - ``full_k/v`` : [1, H_kv, n_k, d]   (pre-GQA-expansion)
and expects a return of shape ``[1, H, 1, d]``.

ssa's ``attn`` uses ``q=[H,d]``, ``K,V=[n_k,H_kv,d]`` — this module converts
between the two, drives reproducible per-(layer, position) RNG, and accumulates
the read statistics used for the bytes-avoided metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from engine.attention import Attention

from ..attn import attn

_SAMPLING_IMPLS = {"santa", "santa_strat", "santa_sys", "santa_hybrid"}


@dataclass
class ReadStats:
    """Accumulates value-row reads across decode steps (per head, averaged).

    ``read_fraction`` is the avoided-bytes metric: fraction of the n_k value rows
    actually read. Dense reads all of them (1.0); samplers read far fewer.
    """

    steps: int = 0
    reads_sum: float = 0.0          # Σ over steps of mean-over-heads reads
    n_k_sum: float = 0.0            # Σ over steps of n_k

    def record(self, *, n_k: int, reads_per_head: torch.Tensor) -> None:
        self.steps += 1
        self.reads_sum += float(reads_per_head.float().mean().item())
        self.n_k_sum += float(n_k)

    @property
    def avg_reads(self) -> float:
        return self.reads_sum / self.steps if self.steps else 0.0

    @property
    def avg_n_k(self) -> float:
        return self.n_k_sum / self.steps if self.steps else 0.0

    @property
    def read_fraction(self) -> float:
        return self.reads_sum / self.n_k_sum if self.n_k_sum else 1.0


def _seed(base_seed: int, layer_idx: int, step: int) -> int:
    return (base_seed * 100_003 + layer_idx * 1_009 + step) % (2**31 - 1)


def make_decode_op(impl: str, *, base_seed: int, stats: ReadStats, cfg: dict):
    """Build a decode-seam callback for ``impl`` (closure owns its step counter)."""
    is_sampling = impl in _SAMPLING_IMPLS
    k_h = int(cfg.get("k_h", 0))
    counter = {"step": 0}

    def op(q, full_k, full_v, *, scale, layer_idx):
        qd = q[0, :, 0, :]                          # [H, d]
        K = full_k[0].permute(1, 0, 2).contiguous() # [n_k, H_kv, d]
        V = full_v[0].permute(1, 0, 2).contiguous()
        n_k = K.shape[0]

        call_cfg = dict(cfg)
        if is_sampling:
            gen = torch.Generator(device=qd.device).manual_seed(
                _seed(base_seed, layer_idx, counter["step"])
            )
            out, info = attn(qd, K, V, impl=impl, generator=gen, return_info=True, **call_cfg)
            # Distinct rows read = effective head keys + unique tail draws, capped
            # at n_k (head ∪ tail can't exceed the cache; when the head absorbs
            # ~all mass the tail is discarded, so reads saturate at n_k).
            reads = (min(k_h, n_k) + info.unique.to(qd.device).float()).clamp_(max=float(n_k))
        else:
            out = attn(qd, K, V, impl=impl, **call_cfg)
            if impl == "topk":                                  # reads exactly the top-k rows
                kk = min(int(cfg.get("k", n_k)), n_k)
                reads = torch.full((qd.shape[0],), float(kk), device=qd.device)
            else:                                               # dense: reads everything
                reads = torch.full((qd.shape[0],), float(n_k), device=qd.device)

        stats.record(n_k=n_k, reads_per_head=reads)
        counter["step"] += 1
        return out.unsqueeze(0).unsqueeze(2)                   # [1, H, 1, d]

    return op


def install(model, impl: str, *, base_seed: int = 0, **cfg) -> ReadStats:
    """Set the decode op on every ``Attention`` module; return a shared ReadStats."""
    stats = ReadStats()
    for m in model.modules():
        if isinstance(m, Attention):
            m.decode_attn_op = make_decode_op(impl, base_seed=base_seed, stats=stats, cfg=cfg)
    return stats


def uninstall(model) -> None:
    """Restore the engine's default (dense SDPA) decode path."""
    for m in model.modules():
        if isinstance(m, Attention):
            m.decode_attn_op = None
