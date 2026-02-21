"""Thread-local attention context for paged attention engine.

Stores per-forward-pass metadata (prefill vs decode, cu_seqlens, slot mappings,
block tables) in thread-local storage for FastAPI threadpool safety.
"""

import threading
from dataclasses import dataclass, field
from typing import Optional

import torch

_tls = threading.local()


@dataclass
class AttentionContext:
    is_prefill: bool = False
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None


def get_context() -> AttentionContext:
    ctx = getattr(_tls, "context", None)
    if ctx is None:
        ctx = AttentionContext()
        _tls.context = ctx
    return ctx


def set_context(
    is_prefill: bool,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: Optional[torch.Tensor] = None,
    context_lens: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
) -> None:
    _tls.context = AttentionContext(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
    )


def reset_context() -> None:
    _tls.context = AttentionContext()
