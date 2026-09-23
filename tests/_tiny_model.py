"""Tiny synthetic Qwen3 model for CPU integration tests (no weights download).

Replicates the random-``LoadedModel`` recipe from ``tests/_tiny.py`` in
github.com/aaholmes/llms so we can exercise the engine's real forward pass +
decode-attention hook on CPU in float32.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from engine.model import Qwen3Model
from engine.weights import LoadedModel


@dataclass(frozen=True)
class TinyCfg:
    vocab_size: int = 48
    hidden_size: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 8
    intermediate_size: int = 64
    max_position_embeddings: int = 64
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0


def tiny_model(c: TinyCfg = TinyCfg(), *, seed: int = 0) -> Qwen3Model:
    torch.manual_seed(seed)
    cfg = PretrainedConfig(
        vocab_size=c.vocab_size, hidden_size=c.hidden_size,
        num_hidden_layers=c.num_hidden_layers, num_attention_heads=c.num_attention_heads,
        num_key_value_heads=c.num_key_value_heads, head_dim=c.head_dim,
        intermediate_size=c.intermediate_size, max_position_embeddings=c.max_position_embeddings,
        rms_norm_eps=c.rms_norm_eps, tie_word_embeddings=True, rope_theta=c.rope_theta,
        attention_bias=False,
    )
    H = c.num_attention_heads * c.head_dim
    KV = c.num_key_value_heads * c.head_dim
    state: dict[str, torch.Tensor] = {
        "embed.weight": torch.randn(c.vocab_size, c.hidden_size),
        "final_norm.weight": torch.ones(c.hidden_size),
    }
    for i in range(c.num_hidden_layers):
        p = f"layers.{i}."
        state[p + "norm1.weight"] = torch.ones(c.hidden_size)
        state[p + "norm2.weight"] = torch.ones(c.hidden_size)
        state[p + "attn.q.weight"] = torch.randn(H, c.hidden_size) * 0.1
        state[p + "attn.k.weight"] = torch.randn(KV, c.hidden_size) * 0.1
        state[p + "attn.v.weight"] = torch.randn(KV, c.hidden_size) * 0.1
        state[p + "attn.o.weight"] = torch.randn(c.hidden_size, H) * 0.1
        state[p + "attn.q_norm.weight"] = torch.ones(c.head_dim)
        state[p + "attn.k_norm.weight"] = torch.ones(c.head_dim)
        state[p + "ffn.gate.weight"] = torch.randn(c.intermediate_size, c.hidden_size) * 0.1
        state[p + "ffn.up.weight"] = torch.randn(c.intermediate_size, c.hidden_size) * 0.1
        state[p + "ffn.down.weight"] = torch.randn(c.hidden_size, c.intermediate_size) * 0.1
    return Qwen3Model.from_loaded(LoadedModel(state=state, config=cfg))
