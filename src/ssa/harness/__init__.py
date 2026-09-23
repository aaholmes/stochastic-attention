"""Measurement harness: Monte-Carlo mean + variance-trace vs sample budget S.

This is the scientific core: it tests unbiasedness and the ~1/S variance decay
that every downstream result depends on.
"""

from __future__ import annotations
