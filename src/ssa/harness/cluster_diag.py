"""Cluster diagnostic — does clustering keys by content work on real attention?

A kernel-free first test: on real Qwen3-4B keys/queries, cluster the
keys (after rotary position embedding, RoPE) and measure whether the cheap per-cluster *summary* is enough to
(a) keep blocks uniformly hot/cold, (b) estimate each cluster's mass (free energy)
well, and (c) prune most clusters while still capturing the attention mass. If these
hold, clustering the key-value cache to skip key reads is worth pursuing; if not,
we've learned it for the price of a forward pass.

All masses are computed in a max-shifted, float64 frame for numerical safety; the
shift cancels in every ratio and in the free-energy *error*.
"""

from __future__ import annotations

import torch

# ---- k-means (Lloyd) --------------------------------------------------------

def kmeans(X: torch.Tensor, k: int, *, iters: int = 25, seed: int = 0):
    """Plain Lloyd k-means. X: [n, d] -> (labels [n], centers [k, d]). float64."""
    X = X.to(torch.float64)
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    centers = X[torch.randperm(n, generator=g)[:k]].clone()
    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        d2 = torch.cdist(X, centers).pow(2)        # [n, k]
        new = d2.argmin(1)
        if torch.equal(new, labels):
            labels = new
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = X[m].mean(0)
            else:  # reseed an empty cluster on the worst-fit point
                centers[c] = X[d2.min(1).values.argmax()]
    return labels, centers


# ---- per-cluster summary (the maintained state) -----------------------------

def cluster_stats(K: torch.Tensor, labels: torch.Tensor, k: int) -> dict:
    """Per-cluster center, diagonal variance, size, radius. Diagonal Σ (cheap/realistic)."""
    K = K.to(torch.float64)
    d = K.shape[1]
    centers = torch.zeros(k, d, dtype=torch.float64)
    diag_var = torch.zeros(k, d, dtype=torch.float64)
    sizes = torch.zeros(k, dtype=torch.float64)
    radii = torch.zeros(k, dtype=torch.float64)
    for c in range(k):
        m = labels == c
        sizes[c] = float(m.sum())
        if not m.any():
            continue
        Kc = K[m]
        centers[c] = Kc.mean(0)
        diag_var[c] = Kc.var(0, unbiased=False)
        radii[c] = (Kc - centers[c]).norm(dim=1).max()
    return {"centers": centers, "diag_var": diag_var, "sizes": sizes, "radii": radii}


# ---- per-query metrics ------------------------------------------------------

def query_metrics(q: torch.Tensor, K: torch.Tensor, labels: torch.Tensor, stats: dict) -> dict:
    """For one query: within-cluster score spread, free-energy estimate error, pruning curve.

    Uses the *exact* within-cluster score variance `Var_j(q·k_j)` (= `qᵀΣ_b q` with the
    full covariance), computed directly from the keys — not a diagonal proxy — so the
    free-energy test measures the Gaussian *form*, not a covariance approximation.
    Pruning is *oracle* (rank by true mass): it isolates clustering quality (does the
    mass land in few blocks) from estimate quality (the separate `fe_err`).
    """
    q = q.to(torch.float64)
    K = K.to(torch.float64)
    k = int(stats["sizes"].shape[0])
    sizes = stats["sizes"]
    s = K @ q                                            # true scores [n]
    shift = float(s.max())
    denom = sizes.clamp_min(1.0)

    # exact per-cluster score mean and variance (full-covariance qᵀΣq, no diagonal proxy)
    sum_s = torch.zeros(k, dtype=torch.float64).index_add_(0, labels, s)
    sum_s2 = torch.zeros(k, dtype=torch.float64).index_add_(0, labels, s * s)
    mu = sum_s / denom
    var = (sum_s2 / denom - mu.pow(2)).clamp_min(0.0)    # within-cluster score variance
    qnorm = float(q.norm())

    w = torch.exp(s - shift)
    true_m = torch.zeros(k, dtype=torch.float64).index_add_(0, labels, w)
    log_true = true_m.clamp_min(1e-300).log()

    # exp-space within-cluster participation ratio: how many tokens carry the mass
    # *inside* a cluster. ≈1 ⇒ max-dominated (one token = the cluster) ⇒ moments can't
    # estimate the mass; ≈|b| ⇒ uniformly hot ⇒ estimable and aggregation pays off.
    sum_w2 = torch.zeros(k, dtype=torch.float64).index_add_(0, labels, w * w)
    within_pr = true_m.pow(2) / sum_w2.clamp_min(1e-300)         # [k], in [1, |b|]
    mass_frac = true_m / true_m.sum().clamp_min(1e-300)
    within_pr_massw = float((mass_frac * within_pr).sum())       # mass-weighted (hot clusters dominate)
    # Gaussian free-energy estimate with the exact variance, in log space (overflow-free)
    log_est = sizes.clamp_min(1e-300).log() + (mu - shift) + 0.5 * var
    log_upper = sizes.clamp_min(1e-300).log() + (mu - shift) + qnorm * stats["radii"]

    # magnitude-based mass estimate: scores ≈ |k_j|·(q·ĉ_b) with ĉ_b the unit cluster
    # direction (tests the "domination is along magnitude, not direction" hypothesis —
    # exact when a cluster's keys share a direction, no Gaussian assumption).
    cdir = stats["centers"] / stats["centers"].norm(dim=1, keepdim=True).clamp_min(1e-12)
    d_b = cdir @ q                                       # [k] per-unit-magnitude score
    kmag = K.norm(dim=1)                                 # [n]
    shat = kmag * d_b[labels]                            # [n] predicted scores
    m_mag = torch.zeros(k, dtype=torch.float64).index_add_(0, labels, torch.exp(shat - shift))
    log_mag = m_mag.clamp_min(1e-300).log()

    live = sizes > 0
    eps = log_true[live] - log_est[live]
    eps_mag = log_true[live] - log_mag[live]

    # The decision-relevant metric: rank clusters by each ranking key, fetch greedily,
    # and track cumulative TRUE mass vs cumulative keys fetched. "oracle" = best possible
    # (rank by true mass); "gauss"/"mag" = what the cheap estimates actually achieve.
    def _curve(rank_key):
        order = torch.argsort(rank_key, descending=True)
        return (sizes[order].cumsum(0) / sizes.sum(),
                true_m[order].cumsum(0) / true_m.sum())

    return {
        "score_std": var.sqrt(),                         # within-cluster score spread (nats)
        "mu_std": float(mu[live].std()),                 # across-cluster spread of mean scores
        "fe_err_abs": float(eps.abs().mean()),           # |ε| nats (Gaussian) — demoted to a side check
        "fe_err_mag": float(eps_mag.abs().mean()),       # |ε| nats (magnitude estimate)
        "upper_ok": bool((log_upper[live] + 1e-9 >= log_true[live]).all()),
        "within_pr": within_pr, "within_pr_massw": within_pr_massw,
        "curves": {"oracle": _curve(true_m), "gauss": _curve(log_est), "mag": _curve(log_mag)},
    }


def _frac_keys_for_mass(keys_cum, mass_cum, target: float) -> float:
    """Fraction of keys fetched to first reach `target` cumulative true mass."""
    idx = int((mass_cum >= target).float().argmax())
    return float(keys_cum[idx])


# ---- summary over many queries, with clustering baselines -------------------

def query_whiten(K: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Map keys into the query metric: `K · C_q^{1/2}`, `C_q = E[q qᵀ]`.

    Euclidean k-means on the result minimizes within-cluster *score* variance
    `tr(C_q Σ_b)` (what the free-energy estimate needs) instead of raw key spread
    `tr(Σ_b)` — it scales up the directions queries actually point and collapses the
    ones no query looks at (a query-aware, Mahalanobis-distance objective).
    """
    Q = Q.to(torch.float64)
    Cq = (Q.t() @ Q) / Q.shape[0]                                # [d, d]
    evals, evecs = torch.linalg.eigh(Cq)
    W = evecs @ torch.diag(evals.clamp_min(0).sqrt()) @ evecs.t()  # C_q^{1/2}
    return K.to(torch.float64) @ W


def _label_schemes(K, B, seed, Q=None):
    n = K.shape[0]
    k = (n + B - 1) // B
    schemes = {"kmeans": kmeans(K, k, seed=seed)[0]}
    Kn = K / K.to(torch.float64).norm(dim=1, keepdim=True).clamp_min(1e-12)
    schemes["kmeans_sphere"] = kmeans(Kn, k, seed=seed)[0]       # cluster by direction only
    if Q is not None:                                            # query-whitened (Mahalanobis)
        schemes["kmeans_qw"] = kmeans(query_whiten(K, Q), k, seed=seed)[0]
    schemes["contiguous"] = (torch.arange(n) // B).clamp(max=k - 1)   # position blocks
    g = torch.Generator().manual_seed(seed)
    schemes["random"] = (torch.randperm(n, generator=g) // B).clamp(max=k - 1)
    return schemes, k


def summarize(K: torch.Tensor, Q: torch.Tensor, *, B: int = 16, target: float = 0.99, seed: int = 0) -> dict:
    """Cluster K (n_k/B clusters), average metrics over the queries Q. Returns a dict."""
    schemes, k = _label_schemes(K, B, seed, Q=Q)
    out = {"n_k": int(K.shape[0]), "n_queries": int(Q.shape[0]), "B": B, "n_clusters": k}
    for name, labels in schemes.items():
        stats = cluster_stats(K, labels, k)
        oracle, gauss, mag, fe, fe_mag, pr, upper_ok = [], [], [], [], [], [], True
        for q in Q:
            m = query_metrics(q, K, labels, stats)
            c = m["curves"]
            oracle.append(_frac_keys_for_mass(*c["oracle"], target))
            gauss.append(_frac_keys_for_mass(*c["gauss"], target))
            mag.append(_frac_keys_for_mass(*c["mag"], target))
            fe.append(m["fe_err_abs"]); fe_mag.append(m["fe_err_mag"])
            pr.append(m["within_pr_massw"])
            upper_ok = upper_ok and m["upper_ok"]
        mean = lambda xs: float(torch.tensor(xs).mean())
        out[name] = {
            "frac_keys_oracle": mean(oracle),        # best possible (rank by true mass)
            "frac_keys_gauss": mean(gauss),          # rank by Gaussian moment estimate
            "frac_keys_mag": mean(mag),              # rank by magnitude estimate (your hypothesis)
            "frac_keys_for_99pct_mass": mean(oracle),  # alias (back-compat)
            "free_energy_err_nats": mean(fe),
            "free_energy_err_mag_nats": mean(fe_mag),
            "within_cluster_participation": mean(pr),
            "radius_bound_valid": upper_ok,
        }
    return out


# ---- real-model capture + entrypoint ----------------------------------------

def _capture(model, ids, *, layer: int, kv_head: int, prefill: int, steps: int, device):
    """Prefill, then capture post-RoPE keys (prefill set) + queries over `steps` decode steps."""
    from engine.attention import Attention

    from ..attn.geometry import attn_weights, gqa_expand, weighted_sum

    attn_mods = [m for m in model.modules() if isinstance(m, Attention)]
    target = attn_mods[layer]
    cap = {"K": None, "Q": []}
    G = target.num_heads // target.num_kv_heads

    def op(q, full_k, full_v, *, scale, layer_idx):
        if cap["K"] is None:
            cap["K"] = full_k[0, kv_head].clone()                       # [n_k, d] prefill keys
        qh = q[0, kv_head * G:(kv_head + 1) * G, 0, :]                  # [G, d] queries this step
        cap["Q"].append(qh.clone())
        qd = q[0, :, 0, :]                                              # dense readout (non-perturbing)
        Kk = full_k[0].permute(1, 0, 2).contiguous()
        Vv = full_v[0].permute(1, 0, 2).contiguous()
        A = attn_weights(qd, Kk)
        out = weighted_sum(A, gqa_expand(Vv, qd.shape[0]))
        return out.unsqueeze(0).unsqueeze(2)

    target.decode_attn_op = op
    cache = model.alloc_cache(ids.shape[1] + 1)
    with torch.inference_mode():
        model(ids[:, :prefill], cache, start_pos=0)
        for t in range(prefill, min(prefill + steps, ids.shape[1])):
            model(ids[:, t:t + 1], cache)
    target.decode_attn_op = None
    K = cap["K"].to(torch.float64)
    Q = torch.cat(cap["Q"], 0).to(torch.float64)
    return K, Q


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from .ppl_sweep import _load_model, _wikitext_chunks
    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 12, 24, 35])
    p.add_argument("--kv-head", type=int, default=0)
    p.add_argument("--prefill", type=int, default=1024)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--block", type=int, nargs="+", default=[16])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = _load_model(args.model, args.device, dtype)
    chunk = _wikitext_chunks(args.model, max_chunks=1, chunk_len=args.prefill + args.steps + 1,
                             device=args.device)[0]

    per_layer = {}
    print(f"{'layer':>5} {'B':>3} {'scheme':>13} {'oracle':>7} {'gauss':>7} {'mag':>7} "
          f"{'within_PR':>9}   (frac keys @ 99% mass, by ranking)")
    for layer in args.layers:
        K, Q = _capture(model, chunk, layer=layer, kv_head=args.kv_head,
                        prefill=args.prefill, steps=args.steps, device=args.device)
        per_block = {}
        for B in args.block:
            s = summarize(K.cpu(), Q.cpu(), B=B)
            per_block[str(B)] = s
            for scheme in ("kmeans", "kmeans_sphere", "kmeans_qw", "random"):
                r = s[scheme]
                print(f"{layer:>5} {B:>3} {scheme:>13} {r['frac_keys_oracle']*100:>6.1f}% "
                      f"{r['frac_keys_gauss']*100:>6.1f}% {r['frac_keys_mag']*100:>6.1f}% "
                      f"{r['within_cluster_participation']:>9.2f}")
        per_layer[str(layer)] = per_block

    payload = stamp({"kind": "cluster_diagnostic", "model": args.model, "block": args.block,
                     "kv_head": args.kv_head, "prefill": args.prefill, "steps": args.steps,
                     "per_layer": per_layer})
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"cluster_diag_{payload.get('git_sha', 'nogit')[:8]}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
