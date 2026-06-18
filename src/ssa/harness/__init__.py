"""Phase A/B measurement harness: Monte-Carlo mean + variance-trace vs S.

This is the scientific core (design doc §0, §7): proves unbiasedness and the
~1/S variance decay that gates everything downstream.
"""

from __future__ import annotations
