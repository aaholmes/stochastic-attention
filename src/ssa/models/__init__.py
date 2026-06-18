"""Integration with the sibling ``../llms`` Qwen3 engine (Phase C).

Wires ssa's swappable attention estimators into the engine's decode seam
(``engine.attention.Attention.decode_attn_op``) without forking it.
"""

from __future__ import annotations
