"""Paged CodePredictor model wrapper — shares all weights from the HF CodePredictor.

Wraps Qwen3TTSTalkerCodePredictorModelForConditionalGeneration layers as
PagedDecoderLayer list. Implements the 15-step codebook generation loop
with its own short-lived paged KV cache (1 block per run, reset between frames).

Config (1.7B model): 5 layers, hidden=1024, 16 heads, 8 KV heads, head_dim=64
"""

from typing import Optional

import torch
from torch import nn

from ..attention import PagedAttention
from .paged_decoder_layer import PagedDecoderLayer
from ..rotary_embedding import get_rope


class PagedPredictorModel(nn.Module):
    """Wraps the HF CodePredictor transformer body with paged attention layers."""

    def __init__(self, hf_predictor_model: nn.Module, config, talker_hidden_size: int):
        """
        Args:
            hf_predictor_model: The HF Qwen3TTSTalkerCodePredictorModel instance
            config: Qwen3TTSTalkerCodePredictorConfig
            talker_hidden_size: Hidden size of the Talker (for codec embedding dim)
        """
        super().__init__()
        self.config = config

        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads

        # Build RoPE for paged layers (standard 1D, not multimodal)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.rotary_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=rope_scaling,
        )

        # Wrap each HF decoder layer
        self.layers = nn.ModuleList([
            PagedDecoderLayer(
                hf_layer=hf_layer,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                hidden_size=config.hidden_size,
                rotary_emb=self.rotary_emb,
            )
            for hf_layer in hf_predictor_model.layers
        ])

        # Share norm and codec embeddings by reference
        self.norm = hf_predictor_model.norm
        self.codec_embedding = hf_predictor_model.codec_embedding

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


class PagedPredictor(nn.Module):
    """Wraps Qwen3TTSTalkerCodePredictorModelForConditionalGeneration with paged attention.

    Shares: lm_head (15), codec_embedding (15), small_to_mtp_projection, norm
    """

    def __init__(self, hf_code_predictor: nn.Module, config, talker_config):
        """
        Args:
            hf_code_predictor: The HF Qwen3TTSTalkerCodePredictorModelForConditionalGeneration
            config: Qwen3TTSTalkerCodePredictorConfig
            talker_config: Qwen3TTSTalkerConfig (for talker hidden size)
        """
        super().__init__()
        self.config = config
        self.talker_hidden_size = talker_config.hidden_size
        self.model = PagedPredictorModel(
            hf_code_predictor.model, config, talker_config.hidden_size
        )

        # Share output heads and projection by reference
        self.lm_head = hf_code_predictor.lm_head  # ModuleList of 15
        self.small_to_mtp_projection = hf_code_predictor.small_to_mtp_projection

    def forward(
        self,
        input_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: projects from talker hidden → predictor hidden, then transformer."""
        input_embeds = self.small_to_mtp_projection(input_embeds)
        return self.model(input_embeds, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        generation_steps: list[int],
    ) -> torch.Tensor:
        """Compute logits for the appropriate codebook at each generation step.

        Args:
            hidden_states: (total_tokens, hidden_size)
            generation_steps: list of ints, one per sequence, indicating which codebook
                             (0-14) this step generates

        Returns:
            logits: (num_seqs, vocab_size) — last-token logits per sequence
        """
        final_logits = []
        hidden_states = hidden_states.view(len(generation_steps), -1, hidden_states.shape[-1])
        for idx, gen_step in enumerate(generation_steps):
            logits = self.lm_head[gen_step](hidden_states[idx])
            final_logits.append(logits.unsqueeze(0))
        final_logits = torch.cat(final_logits, dim=0)
        return final_logits[:, -1, :]

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor],
        generation_steps: list[int],
    ) -> torch.Tensor:
        """Get codec embeddings for the current generation step.

        During prefill: input_embeds is used directly.
        During decode: input_ids are embedded via the step-appropriate codec embedding.
        """
        if input_embeds is not None and input_embeds.shape[1] > 1:
            return input_embeds
        input_embeds_list = []
        for i, ids in enumerate(input_ids):
            input_embeds_list.append(
                self.model.codec_embedding[generation_steps[i] - 1](ids)
            )
        return torch.stack(input_embeds_list)

    def get_codec_embedding(self):
        return self.model.codec_embedding
