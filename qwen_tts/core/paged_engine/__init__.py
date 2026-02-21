"""Paged attention engine for Qwen3-TTS streaming inference.

Provides opt-in paged attention + CUDA graphs as an alternative to the default
HF-based DynamicCache + torch.compile(reduce-overhead) backend.

Usage:
    from qwen_tts.core.paged_engine import PagedEngine

    engine = PagedEngine(hf_model, gpu_memory_utilization=0.3)
    engine.warmup()  # Must run in same thread as inference

    # In stream_generate_pcm paged path:
    token, hidden, seq = engine.prefill(input_embeds, attention_mask)
    while not done:
        token, codec_ids, logits = engine.step(seq, next_embeds)
        # ... emit/decode as usual ...
    engine.cleanup(seq)

Requires: flash-attn >= 2.0, triton >= 2.1, xxhash
"""

from .engine import PagedEngine

__all__ = ["PagedEngine"]
