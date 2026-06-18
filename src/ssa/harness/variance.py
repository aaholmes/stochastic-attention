"""Phase A measurement core: Monte-Carlo mean + variance-trace vs sample budget S.

This is the scientific gate (design doc §7): the ``santa*`` estimators must (a) be
unbiased — their MC mean converges to ``dense`` — and (b) have a variance-trace that
falls as ~1/S (a log-log slope near -1). Everything is accumulated in **float64**
regardless of the estimator's operating dtype, so low-precision roundoff is never
mistaken for estimator bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..attn import attn
from ..attn.dense import dense


def collect_estimates(
    impl: str,
    q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    S: int,
    runs: int,
    base_seed: int = 0,
    **cfg,
) -> torch.Tensor:
    """Run ``impl`` ``runs`` times with distinct seeds. Returns ``[runs, H, d]`` float64."""
    out = []
    for r in range(runs):
        gen = torch.Generator(device=q.device).manual_seed(base_seed + r)
        est = attn(q, K, V, impl=impl, S=S, generator=gen, **cfg)
        out.append(est.to(torch.float64))
    return torch.stack(out, dim=0)


def mc_mean_and_stderr(estimates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-coordinate Monte-Carlo mean and standard error of the mean."""
    runs = estimates.shape[0]
    mean = estimates.mean(dim=0)
    sd = estimates.std(dim=0, unbiased=True)
    return mean, sd / math.sqrt(runs)


def variance_trace(estimates: torch.Tensor) -> float:
    """Trace of the estimator covariance: sum of per-coordinate variance over runs."""
    var = estimates.var(dim=0, unbiased=True)
    return float(var.sum().item())


def fit_slope(S_values: list[int], var_values: list[float]) -> float:
    """Least-squares slope of log(variance) vs log(S)."""
    logS = np.log(np.asarray(S_values, dtype=np.float64))
    logV = np.log(np.asarray(var_values, dtype=np.float64))
    slope, _ = np.polyfit(logS, logV, 1)
    return float(slope)


@dataclass
class SweepResult:
    impl: str
    S_values: list[int]
    var_values: list[float]
    slope: float


def variance_sweep(
    impl: str,
    q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    S_values: list[int],
    runs: int,
    base_seed: int = 0,
    **cfg,
) -> SweepResult:
    """Variance-trace at each S plus the fitted log-log slope."""
    vt = []
    for S in S_values:
        est = collect_estimates(
            impl, q, K, V, S=S, runs=runs, base_seed=base_seed, **cfg
        )
        vt.append(variance_trace(est))
    return SweepResult(impl, list(S_values), vt, fit_slope(S_values, vt))


# --- entrypoint: stamped Phase A report --------------------------------------

def _default_qkv(seed: int) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tiny float64-on-CPU decode geometry for the standalone Phase A report."""
    H, H_kv, d, n_k = 8, 2, 16, 256
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(H, d, generator=gen, dtype=torch.float64)
    K = torch.randn(n_k, H_kv, d, generator=gen, dtype=torch.float64)
    V = torch.randn(n_k, H_kv, d, generator=gen, dtype=torch.float64)
    geom = {"H": H, "H_kv": H_kv, "d": d, "n_k": n_k}
    return geom, q, K, V


def run_phase_a(*, runs: int = 400, seed: int = 0) -> dict:
    """Run the unbiasedness + variance sweeps and return a stamped result dict."""
    from .stamp import stamp

    geom, q, K, V = _default_qkv(seed)
    S_values = [8, 16, 32, 64, 128, 256]
    impls = ["santa", "santa_strat", "santa_sys"]

    ref = dense(q, K, V).to(torch.float64)
    sweeps = {}
    bias = {}
    for impl in impls:
        sw = variance_sweep(impl, q, K, V, S_values=S_values, runs=runs, base_seed=seed)
        sweeps[impl] = {"S": sw.S_values, "var_trace": sw.var_values, "slope": sw.slope}
        est = collect_estimates(impl, q, K, V, S=64, runs=runs, base_seed=seed)
        mean, stderr = mc_mean_and_stderr(est)
        max_z = float(((mean - ref).abs() / stderr.clamp_min(1e-300)).max().item())
        bias[impl] = {"max_abs_z_at_S64": max_z}

    return stamp(
        {
            "phase": "A",
            "runs": runs,
            "seed": seed,
            "geometry": geom,
            "variance_sweeps": sweeps,
            "unbiasedness": bias,
        }
    )


if __name__ == "__main__":
    import json
    from pathlib import Path

    result = run_phase_a()
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    sha = result.get("git_sha", "nogit")[:8]
    path = out_dir / f"phaseA_variance_{sha}.json"
    path.write_text(json.dumps(result, indent=2))
    for impl, sw in result["variance_sweeps"].items():
        print(f"{impl:12s} slope={sw['slope']:+.3f}  "
              f"max|z|={result['unbiasedness'][impl]['max_abs_z_at_S64']:.2f}")
    print(f"wrote {path}")
