"""Tiny synthetic decode-geometry tensors, shared across ssa tests.

Decode geometry (design doc §2): single query ``q=[H, d]``, ``K,V=[n_k, H_kv, d]``
with GQA grouping ``G = H / H_kv``. Leading-underscore module name so pytest does
not try to collect it as a test file (matches ``../llms/tests/_tiny.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Geom:
    """Attention decode geometry. Defaults are tiny for fast CPU tests."""

    H: int = 4          # query heads
    H_kv: int = 2       # kv heads (GQA group size G = H // H_kv)
    d: int = 16         # head dim
    n_k: int = 64       # cached key/value positions

    @property
    def G(self) -> int:
        return self.H // self.H_kv


# Realistic Llama/Qwen decode geometry for the marked GPU sweeps.
LLAMA_GEOM = Geom(H=32, H_kv=8, d=128, n_k=4096)


def make_qkv(
    g: Geom = Geom(),
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random Gaussian ``q=[H,d]``, ``K,V=[n_k,H_kv,d]`` for one decode step.

    float64 by default: the variance/unbiasedness core measures in double
    precision so bf16 roundoff is never mistaken for estimator bias (plan §Numerics).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(g.H, g.d, generator=gen, dtype=torch.float64)
    K = torch.randn(g.n_k, g.H_kv, g.d, generator=gen, dtype=torch.float64)
    V = torch.randn(g.n_k, g.H_kv, g.d, generator=gen, dtype=torch.float64)
    cast = lambda t: t.to(device=device, dtype=dtype)
    return cast(q), cast(K), cast(V)
