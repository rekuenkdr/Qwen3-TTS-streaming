"""Weight-sharing paged decoder layer wrapper.

Wraps an HF Qwen3TTSTalkerDecoderLayer (or CodePredictor decoder layer)
by sharing all weight tensors (q/k/v/o projections, norms, MLP) via
Python references. Only the attention computation is replaced with
PagedAttention (flash_attn_with_kvcache + block tables).
"""

import torch
from torch import nn

from ..attention import PagedAttention


class PagedDecoderLayer(nn.Module):
    """Wraps a single HF decoder layer with paged attention.

    Weight tensors are shared by reference — no copies, no extra VRAM.
    Only the attention forward path changes (paged KV cache + flash_attn).
    """

    def __init__(
        self,
        hf_layer: nn.Module,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        rotary_emb: nn.Module,
    ):
        super().__init__()

        # Share weight references from HF layer's attention
        hf_attn = hf_layer.self_attn
        self.q_proj = hf_attn.q_proj
        self.k_proj = hf_attn.k_proj
        self.v_proj = hf_attn.v_proj
        self.o_proj = hf_attn.o_proj

        # QK normalization (Qwen3-TTS uses RMSNorm on head dim)
        self.q_norm = hf_attn.q_norm
        self.k_norm = hf_attn.k_norm

        # Share MLP and layer norms
        self.mlp = hf_layer.mlp
        self.input_layernorm = hf_layer.input_layernorm
        self.post_attention_layernorm = hf_layer.post_attention_layernorm

        # Config
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size

        # RoPE (shared across all layers)
        self.rotary_emb = rotary_emb

        # Paged attention (k_cache/v_cache assigned by model_runner)
        self.attn = PagedAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=head_dim ** -0.5,
            num_kv_heads=num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Project Q, K, V
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q, k = self.rotary_emb(positions=positions, query=q, key=k)

        # Paged attention
        o = self.attn(q, k, v)
        attn_output = o.flatten(1, -1)
        output = self.o_proj(attn_output)

        hidden_states = residual + output

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
