"""Shared pytest fixtures + markers for the ssa Phase A suite.

Mirrors the device/dtype/marker conventions of the sibling ``../llms`` engine:
session-scoped ``device``/``dtype``, a ``requires_cuda`` marker auto-skipped when
no NVIDIA GPU is present, so the CPU subset runs in CI.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.fixture(scope="session")
def dtype() -> torch.dtype:
    """Operating dtype for estimator ops — bf16 to match the engine (design §0.5)."""
    return torch.bfloat16


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_cuda: needs an NVIDIA GPU with CUDA")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if torch.cuda.is_available():
        return
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "requires_cuda" in item.keywords:
            item.add_marker(skip_cuda)
