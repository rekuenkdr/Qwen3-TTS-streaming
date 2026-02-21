"""Rotary position embeddings for paged attention engine.

Provides both standard 1D RoPE (for CodePredictor) and 3D multimodal RoPE
(for Talker, with temporal/height/width position dimensions).

These operate on flattened (N, H, D) tensors unlike HF's (B, H, L, D) format.
"""

from typing import Optional

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    mrope_interleaved: bool = False,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 3D multimodal RoPE (temporal, height, width) to q and k."""
    mrope_section = [s * 2 for s in mrope_section]
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))],
        dim=-1,
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))],
        dim=-1,
    ).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Standard 1D RoPE for CodePredictor."""

    def __init__(self, rotary_dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        self.max_seq_len_cached = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: (N,) flattened or (B, L)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        batch, seq_len = positions.shape[0], positions.shape[1]
        num_heads = query.shape[1]
        head_size = query.shape[2]

        query_4d = query.view(batch, seq_len, num_heads, head_size).transpose(1, 2)
        key_4d = key.view(batch, seq_len, key.shape[1], head_size).transpose(1, 2)

        with torch.no_grad():
            # Lazy one-time migration: warmup forward moves buffer to CUDA,
            # subsequent calls (including CUDA graph capture) find it already there.
            if self.inv_freq.device != query.device:
                self.inv_freq = self.inv_freq.to(query.device)
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(batch, -1, 1)
            position_ids_expanded = positions[:, None, :].float()
            device_type = query.device.type if query.device.type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos().to(query.dtype)
                sin = emb.sin().to(query.dtype)

        q_embed, k_embed = apply_rotary_pos_emb(query_4d, key_4d, cos, sin, unsqueeze_dim=1)
        return (
            q_embed.transpose(1, 2).reshape(batch * seq_len, num_heads, head_size),
            k_embed.transpose(1, 2).reshape(batch * seq_len, key.shape[1], head_size),
        )


class TalkerRotaryEmbedding(nn.Module):
    """3D RoPE for Talker (temporal, height, width).

    Expects positions of shape (3, batch, seq_len), (batch, seq_len), or (seq_len,).
    For codec generation all 3 dims are identical, so 1D positions are expanded to 3D.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        mrope_section: list[int],
        mrope_interleaved: bool = False,
        attention_scaling: float = 1.0,
    ):
        super().__init__()
        self.head_size = head_size
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        self.attention_scaling = attention_scaling
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalize positions to (3, batch, seq_len)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).unsqueeze(0).expand(3, 1, -1)
        elif positions.ndim == 2:
            positions = positions.unsqueeze(0).expand(3, -1, -1)

        batch, seq_len = positions.shape[1], positions.shape[2]
        num_heads = query.shape[1]

        query_4d = query.view(batch, seq_len, num_heads, self.head_size).transpose(1, 2)
        key_4d = key.view(batch, seq_len, key.shape[1], self.head_size).transpose(1, 2)

        # Lazy one-time migration: warmup forward moves buffer to CUDA,
        # subsequent calls (including CUDA graph capture) find it already there.
        if self.inv_freq.device != query.device:
            self.inv_freq = self.inv_freq.to(query.device)
        inv_freq = self.inv_freq.float()
        position_ids_expanded = positions[:, :, None, :].float()
        inv_freq_expanded = inv_freq[None, None, :, None].expand(3, batch, -1, 1)

        device_type = "cuda" if query.device.type == "cuda" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = (emb.cos() * self.attention_scaling).to(query.dtype)
            sin = (emb.sin() * self.attention_scaling).to(query.dtype)

        q_embed, k_embed = apply_multimodal_rotary_pos_emb(
            query_4d, key_4d, cos, sin,
            self.mrope_section,
            self.mrope_interleaved,
            unsqueeze_dim=1,
        )
        return (
            q_embed.transpose(1, 2).reshape(batch * seq_len, num_heads, self.head_size),
            k_embed.transpose(1, 2).reshape(batch * seq_len, key.shape[1], self.head_size),
        )


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Optional[dict] = None,
) -> nn.Module:
    if rope_scaling is not None and "mrope_section" in rope_scaling:
        return TalkerRotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position,
            base=base,
            mrope_section=rope_scaling["mrope_section"],
            mrope_interleaved=rope_scaling.get("interleaved", False),
            attention_scaling=rope_scaling.get("attention_scaling", 1.0),
        )
    return RotaryEmbedding(
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position,
        base=base,
    )
