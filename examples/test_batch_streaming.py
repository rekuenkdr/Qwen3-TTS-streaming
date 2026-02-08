"""Test batch streaming voice clone generation.

Generates audio for multiple texts (potentially with different voices) in a
single batched pass through the transformer. All items advance in lockstep.
"""

import time

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


def log_time(start, operation):
    elapsed = time.time() - start
    print(f"[{elapsed:.2f}s] {operation}")
    return time.time()


total_start = time.time()

# Load model
start = time.time()
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
start = log_time(start, "Load Base model")

# Create a voice clone prompt (using the same voice for all items in this test)
ref_audio_path = "kuklina-1.wav"  # Replace with your reference audio
ref_text = (
    "Это брат Кэти, моей одноклассницы. А что у тебя с рукой? И почему ты голая? "
    "У него ведь куча наград по боевым искусствам."
)

voice_prompt = model.create_voice_clone_prompt(
    ref_audio=ref_audio_path,
    ref_text=ref_text,
)
start = log_time(start, "Create voice clone prompt")

# Batch items: different texts, same voice (broadcast)
texts = [
    "Hello! This is the first batch item with a short sentence.",
    "And this is the second batch item. It has a bit more text to synthesize.",
    "Third item here. Testing batch streaming with multiple items at once!",
]
languages = ["English", "English", "English"]

# ============== Batch Streaming ==============
print(f"\n--- Batch streaming ({len(texts)} items) ---")
start = time.time()

# Accumulate per-item chunks
item_chunks: list[list[np.ndarray]] = [[] for _ in range(len(texts))]
chunk_count = 0
sr = 24000

for chunks_list, chunk_sr in model.batch_stream_generate_voice_clone(
    text=texts,
    language=languages,
    voice_clone_prompt=voice_prompt,
    emit_every_frames=8,
    decode_window_frames=80,
    overlap_samples=512,
    max_frames=8000,
    first_chunk_emit_every=5,
    first_chunk_decode_window=48,
    first_chunk_frames=48,
):
    sr = chunk_sr
    chunk_count += 1
    sizes = []
    for b, chunk in enumerate(chunks_list):
        if chunk.size > 0:
            item_chunks[b].append(chunk)
            sizes.append(f"item{b}={chunk.size}")
        else:
            sizes.append(f"item{b}=empty")
    if chunk_count <= 5 or chunk_count % 10 == 0:
        print(f"  Chunk {chunk_count}: {', '.join(sizes)}")

start = log_time(start, f"Batch streaming done ({chunk_count} chunks)")

# Save per-item outputs
for i, chunks in enumerate(item_chunks):
    if chunks:
        combined = np.concatenate(chunks)
        filename = f"batch_item_{i}.wav"
        sf.write(filename, combined, sr)
        duration_ms = len(combined) / sr * 1000
        print(f"  Saved {filename}: {duration_ms:.0f}ms, {len(combined)} samples")
    else:
        print(f"  Item {i}: no audio generated")

# ============== Compare: Sequential single-item streaming ==============
print(f"\n--- Sequential single-item streaming ({len(texts)} items) ---")
start = time.time()

for i, text in enumerate(texts):
    item_single_chunks = []
    for chunk, chunk_sr in model.stream_generate_voice_clone(
        text=text,
        language=languages[i],
        voice_clone_prompt=voice_prompt,
        emit_every_frames=8,
        decode_window_frames=80,
        overlap_samples=512,
        max_frames=8000,
        first_chunk_emit_every=5,
        first_chunk_decode_window=48,
        first_chunk_frames=48,
    ):
        if chunk.size > 0:
            item_single_chunks.append(chunk)

    if item_single_chunks:
        combined = np.concatenate(item_single_chunks)
        filename = f"single_item_{i}.wav"
        sf.write(filename, combined, chunk_sr)
        duration_ms = len(combined) / chunk_sr * 1000
        print(f"  Saved {filename}: {duration_ms:.0f}ms")

start = log_time(start, "Sequential streaming done")

total_elapsed = time.time() - total_start
print(f"\nTotal time: {total_elapsed:.2f}s")
