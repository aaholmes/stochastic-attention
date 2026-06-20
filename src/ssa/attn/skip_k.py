"""`skip_k` — magnitude-ranked cluster selection (the end-to-end skip-K test, §3.10).

At decode, instead of reading all keys, cluster the keys by *direction* (spherical
k-means), rank clusters by the **magnitude estimate** `m̂_b = Σ_{j∈b} e^{|k_j|(q·ĉ_b)/√d}`
(read from per-key magnitudes + per-cluster unit center — no full key reads), select
the top clusters covering `coverage` of the estimated mass, and compute attention
**only over the selected clusters' keys/values**.

This is the *biased* (deterministic top-cluster) version — it drops the un-selected
tail rather than importance-correcting it — so it measures the *ceiling*: does
magnitude-selected attention preserve perplexity at a small read fraction? The
unbiased variant (selected head + IS-sampled tail) is the follow-up.

`coverage → 1` selects every cluster ⇒ exact `dense`. The read count is the number
of selected keys (full K/V fetched); the magnitude scalars used for selection add
~`1/d` per key on top (reported separately in the write-up, not here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import register


@dataclass
class SkipInfo:
    idx: torch.Tensor       # placeholder (per-head selection differs)
    unique: torch.Tensor    # [H] keys actually read (selected)


def _spherical_kmeans(Kn: torch.Tensor, k: int, *, iters: int = 12, seed: int = 0):
    """Cluster unit vectors by cosine similarity. Kn: [n, d] (unit rows) -> (labels, unit centers)."""
    n, d = Kn.shape
    g = torch.Generator(device=Kn.device).manual_seed(seed)
    c = Kn[torch.randperm(n, generator=g, device=Kn.device)[:k]].clone()
    labels = torch.full((n,), -1, dtype=torch.long, device=Kn.device)
    for _ in range(iters):
        new = (Kn @ c.t()).argmax(1)
        if torch.equal(new, labels):
            break
        labels = new
        csum = torch.zeros(k, d, dtype=Kn.dtype, device=Kn.device).index_add_(0, labels, Kn)
        c = csum / csum.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return labels, c


@register("skip_k")
def skip_k(q, K, V, *, coverage: float = 0.99, B: int = 16, layer_idx: int = 0,
           generator=None, return_info: bool = False, **cfg):
    """Attention over magnitude-selected clusters. Deterministic (generator ignored)."""
    H, d = q.shape
    n_k, H_kv, _ = K.shape
    G = H // H_kv
    scale = 1.0 / math.sqrt(d)
    out = torch.empty(H, d, dtype=V.dtype, device=V.device)
    reads = torch.zeros(H, dtype=torch.long, device=q.device)
    k = max(1, (n_k + B - 1) // B)

    for hkv in range(H_kv):
        Kk = K[:, hkv, :]
        Vk = V[:, hkv, :]
        kf = Kk.to(torch.float32)
        kmag = kf.norm(dim=1)                                    # [n_k]  per-key magnitude
        Kn = kf / kmag.clamp_min(1e-12).unsqueeze(1)
        labels, cdir = _spherical_kmeans(Kn, k, seed=layer_idx * 131 + hkv)

        for gi in range(G):
            h = hkv * G + gi
            qh = q[h].to(torch.float32)
            d_b = (cdir @ qh) * scale                            # [k] per-unit-magnitude scaled score
            shat = kmag * d_b[labels]                            # [n_k] estimated scaled scores
            mhat = torch.zeros(k, dtype=torch.float32, device=q.device).index_add_(
                0, labels, torch.exp(shat - float(shat.max())))
            order = torch.argsort(mhat, descending=True)
            cum = mhat[order].cumsum(0) / mhat.sum().clamp_min(1e-30)
            n_sel = int((cum < coverage).sum().item()) + 1       # clusters to reach coverage
            sel = order[:n_sel]
            mask = torch.zeros(k, dtype=torch.bool, device=q.device)
            mask[sel] = True
            idx = mask[labels].nonzero(as_tuple=True)[0]         # selected key indices

            s = (Kk[idx].to(torch.float32) @ qh) * scale
            A = torch.softmax(s, dim=-1)
            out[h] = (A.to(Vk.dtype) @ Vk[idx])
            reads[h] = idx.numel()

    if return_info:
        return out, SkipInfo(idx=torch.empty(0, dtype=torch.long), unique=reads)
    return out
