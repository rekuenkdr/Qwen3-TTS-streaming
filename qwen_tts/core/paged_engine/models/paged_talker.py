"""Paged Talker model wrapper — shares all weights from the HF Talker model.

Wraps Qwen3TTSTalkerForConditionalGeneration layers as PagedDecoderLayer list.
No weight duplication — only new allocations are the PagedAttention modules
(which just hold references to paged KV cache tensors assigned later).

Config (1.7B model): 28 layers, hidden=2048, 16 heads, 8 KV heads, head_dim=128
"""

import torch
from torch import nn

from ..attention import PagedAttention
from .paged_decoder_layer import PagedDecoderLayer
from ..rotary_embedding import get_rope


class PagedTalkerModel(nn.Module):
    """Wraps the HF Talker transformer body with paged attention layers."""

    def __init__(self, hf_talker_model: nn.Module, config):
        """
        Args:
            hf_talker_model: The HF Qwen3TTSTalkerModel instance (talker.model)
            config: Qwen3TTSTalkerConfig
        """
        super().__init__()
        self.config = config

        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads

        # Build RoPE for paged layers (TalkerRotaryEmbedding — 3D multimodal)
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        self.rotary_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=rope_scaling,
        )

        # Wrap each HF decoder layer with paged attention
        self.layers = nn.ModuleList([
            PagedDecoderLayer(
                hf_layer=hf_layer,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                hidden_size=config.hidden_size,
                rotary_emb=self.rotary_emb,
            )
            for hf_layer in hf_talker_model.layers
        ])

        # Share norm and embeddings by reference
        self.norm = hf_talker_model.norm
        self.codec_embedding = hf_talker_model.codec_embedding
        self.text_embedding = hf_talker_model.text_embedding

    def forward(
        self,
        input_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = input_embeds
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class PagedTalker(nn.Module):
    """Wraps Qwen3TTSTalkerForConditionalGeneration with paged attention.

    Shares: codec_head, text_projection, codec_embedding, text_embedding, norm
    """

    def __init__(self, hf_talker: nn.Module, config):
        """
        Args:
            hf_talker: The HF Qwen3TTSTalkerForConditionalGeneration instance
            config: Qwen3TTSTalkerConfig
        """
        super().__init__()
        self.config = config
        self.model = PagedTalkerModel(hf_talker.model, config)

        # Share output head and text projection by reference
        self.codec_head = hf_talker.codec_head
        self.text_projection = hf_talker.text_projection

    def forward(
        self,
        input_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning hidden states (before logit projection)."""
        return self.model(input_embeds, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to codec vocabulary logits."""
        return self.codec_head(hidden_states)

    def get_input_embeddings(self):
        return self.model.codec_embedding

    def get_text_embeddings(self):
        return self.model.text_embedding
