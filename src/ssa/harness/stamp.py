"""Result-file provenance stamping (design doc §6).

Every result dict gets a header: git SHA, GPU name + driver, torch/CUDA versions.
Callers fold in their own seeds and exact tensor shapes. Reproducibility is the
whole point of a validation prototype.
"""

from __future__ import annotations

import platform
import subprocess

import torch


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "nogit"


def _gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"name": None, "capability": None, "driver": None}
    return {
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "driver": getattr(torch.version, "cuda", None),
    }


def stamp(payload: dict) -> dict:
    """Return ``payload`` merged under a reproducibility header."""
    return {
        "git_sha": _git_sha(),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gpu": _gpu_info(),
        **payload,
    }
