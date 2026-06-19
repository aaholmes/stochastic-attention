"""Cluster diagnostic — does the free-energy clustering idea (§3.9) work on real attention?

The decisive, kernel-free Step-0: on real Qwen3-4B keys/queries, cluster the
(post-RoPE) keys and measure whether the cheap per-cluster *summary* is enough to
(a) keep blocks uniformly hot/cold, (b) estimate each cluster's mass (free energy)
well, and (c) prune most clusters while still capturing the attention mass. If these
hold, the clustered-KV / skip-K direction is alive; if not, we've learned it for the
price of a forward pass.

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
    """For one query: within-cluster score spread, free-energy estimate error, pruning curve."""
    q = q.to(torch.float64)
    K = K.to(torch.float64)
    k = stats["sizes"].shape[0]
    s = K @ q                                       # true scores [n]
    shift = float(s.max())

    mu = stats["centers"] @ q                        # [k] mean score per cluster
    qSq = (stats["diag_var"] * q.pow(2)).sum(1)      # [k] within-cluster score variance qᵀΣq
    qnorm = q.norm()

    # true mass per cluster (max-shifted), and the two estimates
    w = torch.exp(s - shift)
    true_m = torch.zeros(k, dtype=torch.float64).index_add_(0, labels, w)  # [k]
    moment_m = stats["sizes"] * torch.exp(mu - shift + 0.5 * qSq)          # Gaussian estimate
    upper_m = stats["sizes"] * torch.exp(mu - shift + qnorm * stats["radii"])  # radius bound

    live = stats["sizes"] > 0
    eps = (true_m[live].clamp_min(1e-300).log() - moment_m[live].clamp_min(1e-300).log())

    # pruning: fetch clusters ranked by the realistic moment estimate; cumulative TRUE mass
    order = torch.argsort(moment_m, descending=True)
    keys_cum = stats["sizes"][order].cumsum(0) / stats["sizes"].sum()
    mass_cum = true_m[order].cumsum(0) / true_m.sum()

    return {
        "score_std": qSq.clamp_min(0).sqrt(),       # per-cluster within-cluster score spread (nats)
        "mu_std": float(mu.std()),                  # across-cluster spread of mean scores
        "fe_err_abs": float(eps.abs().mean()),      # free-energy estimate error |ε| (nats)
        "fe_err_signed": float(eps.mean()),
        "upper_ok": bool((upper_m + 1e-9 >= true_m).all()),  # radius bound is a true upper bound
        "keys_cum": keys_cum, "mass_cum": mass_cum,
    }


def _frac_keys_for_mass(keys_cum, mass_cum, target: float) -> float:
    """Fraction of keys fetched to first reach `target` cumulative true mass."""
    idx = int((mass_cum >= target).float().argmax())
    return float(keys_cum[idx])


# ---- summary over many queries, with clustering baselines -------------------

def _label_schemes(K, B, seed):
    n = K.shape[0]
    k = (n + B - 1) // B
    km_labels, _ = kmeans(K, k, seed=seed)
    contig = (torch.arange(n) // B).clamp(max=k - 1)            # §3.8 position blocks
    g = torch.Generator().manual_seed(seed)
    rand = (torch.randperm(n, generator=g) // B).clamp(max=k - 1)
    return {"kmeans": km_labels, "contiguous": contig, "random": rand}, k


def summarize(K: torch.Tensor, Q: torch.Tensor, *, B: int = 16, target: float = 0.99, seed: int = 0) -> dict:
    """Cluster K (n_k/B clusters), average metrics over the queries Q. Returns a dict."""
    schemes, k = _label_schemes(K, B, seed)
    out = {"n_k": int(K.shape[0]), "n_queries": int(Q.shape[0]), "B": B, "n_clusters": k}
    for name, labels in schemes.items():
        stats = cluster_stats(K, labels, k)
        fracs, fe, std_ratio, upper_ok = [], [], [], True
        for q in Q:
            m = query_metrics(q, K, labels, stats)
            fracs.append(_frac_keys_for_mass(m["keys_cum"], m["mass_cum"], target))
            fe.append(m["fe_err_abs"])
            std_ratio.append(float(m["score_std"].mean()) / (m["mu_std"] + 1e-12))
            upper_ok = upper_ok and m["upper_ok"]
        out[name] = {
            "frac_keys_for_99pct_mass": float(torch.tensor(fracs).mean()),
            "free_energy_err_nats": float(torch.tensor(fe).mean()),
            "within_over_between_score_spread": float(torch.tensor(std_ratio).mean()),
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
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = _load_model(args.model, args.device, dtype)
    chunk = _wikitext_chunks(args.model, max_chunks=1, chunk_len=args.prefill + args.steps + 1,
                             device=args.device)[0]

    per_layer = {}
    print(f"{'layer':>6} {'scheme':>12} {'frac_keys@99%':>14} {'fe_err_nats':>12} {'within/between':>15}")
    for layer in args.layers:
        K, Q = _capture(model, chunk, layer=layer, kv_head=args.kv_head,
                        prefill=args.prefill, steps=args.steps, device=args.device)
        s = summarize(K.cpu(), Q.cpu(), B=args.block)
        per_layer[str(layer)] = s
        for scheme in ("kmeans", "contiguous", "random"):
            r = s[scheme]
            print(f"{layer:>6} {scheme:>12} {r['frac_keys_for_99pct_mass']*100:>13.1f}% "
                  f"{r['free_energy_err_nats']:>12.3f} {r['within_over_between_score_spread']:>15.3f}")

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
