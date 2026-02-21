# Qwen3-TTS Streaming

Real-time streaming audio generation for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

## Features

From [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming):
- `stream_generate_voice_clone()` - streaming with voice cloning
- `stream_generate_pcm()` - real-time PCM audio streaming
- `torch.compile` + CUDA graphs optimization

Added in this fork:
- **Two-phase streaming** - faster first-chunk latency
- **Hann window crossfade** - click-free chunk boundaries with proper fade-in/fade-out
- **Multiple EOS token detection** - broader termination coverage for reliable generation stopping. Fixes sped-up audio and runaway generation in streaming
- **Repetition penalty for streaming** - prevents token loops that cause looping audio. Defaults to 1.0 (disabled) because streaming generates frame-by-frame with CUDA graph constraints where repetition manifests differently than the non-streaming path.

### Experimental (wip/experimental branch)

- **`generate_fast()` codebook predictor** - bypasses HuggingFace `generate()` overhead for the 31-step autoregressive codebook loop. 1.13x faster per-frame in isolation
- **Manual CUDA graph capture for codebook predictor** - captures the full 31-step codebook loop as a single CUDA graph replay. **2.15x faster** per-frame in isolation (12.94ms vs 27.88ms baseline)
- **GPU-resident repetition penalty** - penalty computation on GPU without CPU roundtrip
- **Batch streaming** - process multiple texts in a single batched transformer pass with `batch_stream_generate_voice_clone()`, with per-item state management and independent EOS detection
- **Batch compaction** - removes finished items from GPU tensors mid-batch
- **Async CUDA stream decoding** - overlaps AR generation with speech decoding on a separate CUDA stream (disabled by default, no speedup observed on single GPU)
- **Paged attention engine** — opt-in alternative backend using flash-attn with paged KV cache, prefix caching via content hashing, and CUDA graph capture for decode steps

## Installation

```bash
sudo apt install sox
pip install torch torchaudio flash-attn
pip install -e .
```

## Usage

```python
import torch
import sounddevice as sd
from qwen_tts import Qwen3TTSModel

# Load model
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-Base",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# Enable optimizations (recommended for streaming)
model.enable_streaming_optimizations(
    decode_window_frames=80,
    use_compile=True,
    compile_mode="reduce-overhead",
)

# Create voice clone prompt from reference audio
prompt = model.create_voice_clone_prompt(
    ref_audio="reference.wav",
    ref_text="Transcript of the reference audio.",
)

# Stream audio with two-phase settings
for chunk, sr in model.stream_generate_voice_clone(
    text="Hello, this is a streaming TTS demo!",
    language="en",
    voice_clone_prompt=prompt,
    # Phase 2 settings (stable)
    emit_every_frames=12,
    decode_window_frames=80,
    # Phase 1 settings (fast first chunk)
    first_chunk_emit_every=5,
    first_chunk_decode_window=48,
    first_chunk_frames=48,
):
    sd.play(chunk, sr)
    sd.wait()
```

## Batch Streaming

Generate audio for multiple texts in a single batched pass through the transformer. All items advance in lockstep, sharing the KV cache. A single voice prompt can be broadcast to all items, or you can pass one per item.

```python
import numpy as np
import soundfile as sf

# Batch of texts (same voice prompt broadcast to all)
texts = [
    "First sentence to synthesize.",
    "Second sentence, different text.",
    "Third sentence in the batch.",
]

# Accumulate per-item chunks
item_chunks = [[] for _ in range(len(texts))]

for chunks_list, sr in model.batch_stream_generate_voice_clone(
    text=texts,
    language="English",              # broadcast to all items
    voice_clone_prompt=prompt,       # broadcast to all items
    emit_every_frames=8,
    decode_window_frames=80,
    first_chunk_emit_every=5,
    first_chunk_decode_window=48,
    first_chunk_frames=48,
):
    for i, chunk in enumerate(chunks_list):
        if chunk.size > 0:
            item_chunks[i].append(chunk)

# Save each item
for i, chunks in enumerate(item_chunks):
    if chunks:
        sf.write(f"output_{i}.wav", np.concatenate(chunks), sr)
```

Items finish independently (per-item EOS detection), but the generator keeps yielding until all items are done. Finished items receive empty arrays.

## Streaming Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emit_every_frames` | 8 | Emit audio every N frames |
| `decode_window_frames` | 80 | Decoder context window |
| `overlap_samples` | 512 | Crossfade overlap between chunks (0 to disable) |
| `max_frames` | 10000 | Maximum codec frames to generate |
| `first_chunk_emit_every` | 0 | Phase 1 emit interval (0 = disabled) |
| `first_chunk_decode_window` | 48 | Phase 1 decode window |
| `first_chunk_frames` | 48 | Switch to phase 2 after N frames |
| `repetition_penalty` | 1.0 | Penalizes repeated tokens (1.0 = disabled) |
| `repetition_penalty_window` | 100 | Only penalize tokens from the last N steps (0 = unlimited) |

## Two-Phase Streaming

Standard streaming with Qwen's TTS library waits for `emit_every_frames` (e.g., 12) before emitting the first audio. Two-phase uses aggressive settings for the first chunk to improve latency, then switches to stable settings.

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 (First N frames)      │  PHASE 2 (Rest of audio)      │
│  - emit_every = 5 (fast)       │  - emit_every = 12 (stable)   │
│  - decode_window = 48          │  - decode_window = 80         │
│  → FAST first chunk            │  → QUALITY for rest           │
└─────────────────────────────────────────────────────────────────┘
```

### Benchmarks

| Test | Method | emit | 1st Chunk | 1st Spdup | Total | Tot Spdup | RTF |
|------|--------|------|-----------|-----------|-------|-----------|-----|
| 2 | Baseline (no opt) | 12 | 570ms | 1.00x | 3.16s | 1.00x | 0.56 |
| 3 | Optimized | 12 | 389ms | 1.47x | 2.37s | 1.34x | 0.37 |
| 4 | Optimized_2 (stable) | 12 | 382ms | 1.49x | 2.27s | 1.39x | 0.36 |
| 5 | **Two-phase (5→12)** | 5→12 | **208ms** | **2.75x** | 2.58s | 1.23x | 0.39 |

User hears audio **362ms earlier** vs baseline, **174ms earlier** vs only optimized.

**First-chunk latency improvement:**
- vs Baseline: **2.75x faster** (570ms → 208ms, saves 362ms)
- vs Optimized: **1.87x faster** (389ms → 208ms, saves 181ms)
- vs Optimized_2: **1.84x faster** (382ms → 208ms, saves 174ms)

## Codebook Predictor Optimization (Experimental)

The codebook predictor runs 31 sequential autoregressive steps per codec frame (16 code groups - 1 = 15 decode steps after prefill). This is the single biggest per-frame bottleneck in streaming TTS. Three optimization levels were benchmarked using `examples/profile_talker.py`:

### Microbenchmarks (per-frame, 1.7B model)

| Method | Per-Frame | Per-Step | Speedup |
|--------|-----------|----------|---------|
| HF `generate()` (baseline) | 27.88ms | ~1.86ms | 1.0x |
| HF `generate()` (no hidden_states) | 28.24ms | ~1.88ms | ~1.0x |
| `generate_fast()` | 24.59ms | ~1.64ms | **1.13x** |
| `generate_fast()` + stacked weights | 24.14ms | ~1.61ms | 1.16x |
| **`generate_fast()` + CUDA graph** | **12.94ms** | **~0.86ms** | **2.15x** |
| Single decode step (reference) | 1.51ms | 1.51ms | — |

### Projected Inter-Chunk Latency (emit_every=12)

| Method | 12 frames | Improvement |
|--------|-----------|-------------|
| HF `generate()` | ~335ms | baseline |
| `generate_fast()` | ~295ms | -40ms |
| CUDA graph | **~155ms** | **-180ms** |

### What Each Level Does

**`generate_fast()`** replaces HuggingFace's `generate()` with a tight loop that:
- Skips GenerationMixin overhead (stopping criteria, output processing, etc.)
- Uses StaticCache with pre-allocated KV buffers
- Runs sampling inline (top-k + multinomial)
- Eliminates per-step Python allocations

**CUDA graph capture** (`capture_codebook_cuda_graph()`) records the full 31-step `generate_fast()` loop as a single CUDA graph:
- Eliminates all CPU-to-GPU kernel launch overhead
- One `graph.replay()` replaces 31 individual kernel launches
- 2.15x speedup in isolation

## Async CUDA Stream Decoding (Experimental)

> **Note:** `async_decode` is disabled by default (`False`) and not recommended for production use.
> Benchmarks show it provides no speedup — 0.95x for B=1 (single stream) and 0.89x for B>1 (batch)
> on a single GPU. The GPU cannot truly parallelize AR and decode on the same device, so stream
> management overhead outweighs any theoretical overlap benefit. The feature is retained for
> experimentation on multi-GPU setups where AR and decode could run on separate devices.

Without async decode, each streaming step is serial: the AR model generates codec tokens, then blocks while the speech decoder converts them to audio. The AR model sits idle during decode.

With `async_decode=True`, speech decoding runs on a separate CUDA stream. After accumulating enough codec frames, the decode is launched non-blocking on the decode stream while the AR model immediately continues generating the next tokens on the default stream. A CUDA event signals when the decode finishes, and the result is yielded at the start of the next emit cycle.

```
Without async (serial):

  Default Stream: [AR tokens]──[Decode]──wait──[AR tokens]──[Decode]──wait──
                                         ^^^^                        ^^^^
                                         idle                        idle

With async_decode=True (pipelined):

  Default Stream: [AR tokens]──[AR tokens]──[AR tokens]──[AR tokens]──
                        │            │            │
                   event│       event│       event│
                        ▼            ▼            ▼
  Decode Stream:   [  Decode  ]──[  Decode  ]──[  Decode  ]──
                             │            │
                        yield│       yield│
                             ▼            ▼
  Output:               chunk 1      chunk 2      ...
```

## Paged Attention Engine (Experimental)

A vLLM-inspired paged attention backend that replaces HuggingFace's DynamicCache for streaming inference. Uses flash-attn for both prefill (`flash_attn_varlen_func`) and decode (`flash_attn_with_kvcache`), a paged KV cache with 256-token blocks, prefix caching via xxhash content hashing, and CUDA graph capture for decode steps.

```
PagedEngine
├── ModelRunner (prefill/decode, CUDA graph capture/replay)
│   ├── PagedTalker (weight-sharing wrapper, PagedDecoderLayer)
│   └── PagedPredictor (weight-sharing wrapper, 15 codebooks)
├── TalkerScheduler / PredictorScheduler (continuous batching)
├── BlockManager (paged KV allocation, prefix hash cache)
└── AttentionContext (thread-local per-forward metadata)
```

### Usage

```python
# Enable with paged engine
model.enable_streaming_optimizations(use_paged_engine=True)

# Warm up CUDA graphs for decode (recommended)
model.warmup_paged_engine()

# Stream as usual — paged engine is used automatically
for chunk, sr in model.stream_generate_voice_clone(
    text="Hello, streaming with paged attention!",
    language="en",
    voice_clone_prompt=prompt,
    use_paged_engine=True,
):
    sd.play(chunk, sr)
    sd.wait()
```

### Key Features

- Flash-attn paged attention (varlen prefill + kvcache decode)
- 256-token block KV cache with ref-counted prefix caching (xxhash)
- CUDA graph capture for decode at batch sizes [1, 2, 4, 8, ...]
- Weight sharing — no model copies, wraps existing HF weights
- Thread-safe scheduling (threading.Lock for FastAPI)

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_paged_engine` | False | Enable paged attention backend |
| `paged_gpu_memory_utilization` | 0.3 | Fraction of GPU VRAM for KV cache |
| `paged_enforce_eager` | False | Skip CUDA graph capture (debug) |

### Dependencies

Requires `flash-attn`, `triton`, and `xxhash` (not installed by default):

```bash
pip install flash-attn triton xxhash
```

## Audio Quality Fixes

Streaming TTS can produce clicks, pops, and artifacts at chunk boundaries. This fork implements several fixes:

### Crossfade Blending

Chunks are blended using a Hann window crossfade to eliminate boundary discontinuities:

```python
# ~21ms at 24kHz, matches RMS check window
# Lower values may cause clicks, set to 0 to disable
DEFAULT_BLEND_SAMPLES = 512

# Hann crossfade
fade_out = 0.5 * (1 + np.cos(np.pi * t))
fade_in = 0.5 * (1 - np.cos(np.pi * t))
blended = prev_tail * fade_out + curr_head * fade_in
```

### Overlap Trimming

Each chunk is processed in this order to prevent audio duplication (echo artifacts):

1. Crossfade current chunk's HEAD with previous chunk's saved TAIL
2. Apply fade-in (first chunk only)
3. Save FULL processed chunk for next iteration's crossfade
4. Trim END of chunk before emission (this region will be replaced by next chunk's crossfade)
5. Yield trimmed chunk

### First/Last Chunk Fades

- **First chunk**: Hann fade-in prevents pop at audio start
- **Final chunk**: Hann fade-out prevents pop at audio end

## Optimization API

### enable_streaming_optimizations()

Call after loading the model to enable torch.compile and CUDA graphs:

```python
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# Enable optimizations (recommended)
model.enable_streaming_optimizations(
    decode_window_frames=80,         # Must match streaming parameter
    use_compile=True,                # torch.compile the decoder
    compile_mode="reduce-overhead",  # Includes CUDA graphs automatically
)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `decode_window_frames` | 80 | Window size (must match streaming call) |
| `use_compile` | True | Apply torch.compile to decoder |
| `use_cuda_graphs` | True | Capture CUDA graphs for fixed window |
| `compile_mode` | "reduce-overhead" | torch.compile mode |
| `use_fast_codebook` | False | Use `generate_fast()` for codebook (experimental, 1.13x) |
| `compile_codebook_predictor` | True | Apply torch.compile to codebook predictor |
| `use_codebook_cuda_graph` | False | Manual CUDA graph for codebook (experimental, 2.15x isolated but conflicts with reduce-overhead) |
| `use_paged_engine` | False | Enable paged attention backend (experimental, requires flash-attn) |
| `paged_gpu_memory_utilization` | 0.3 | Fraction of GPU VRAM allocated for paged KV cache |
| `paged_enforce_eager` | False | Skip CUDA graph capture for paged engine (debug) |

## Profiling

Run the codebook predictor microbenchmarks:

```bash
cd examples
python profile_talker.py
```

Measures all codebook generation paths (HF generate, generate_fast, CUDA graph capture) and runs a torch.profiler trace over 5 streaming chunks.

---

Based on:
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming)
- [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (paged attention engine reference)
