"""Integration with the Qwen3 inference engine from github.com/aaholmes/llms.

Wires ssa's swappable attention estimators into the engine's decode-attention hook
(``engine.attention.Attention.decode_attn_op``) without forking it.
"""

from __future__ import annotations
