"""Semi-stochastic sparse attention — the statistical core.

A self-contained, model-free package implementing the swappable ``attn`` interface
and the harness that tests every estimator for unbiasedness and ~1/S variance decay.
"""

from __future__ import annotations
