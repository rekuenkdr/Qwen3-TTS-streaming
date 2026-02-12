"""
Profile Talker forward pass to identify bottlenecks.

Measures:
1. 31 sequential raw model.forward() calls (matching actual codebook count)
2. HF generate() with output_hidden_states=True vs False
3. Sampling overhead in isolation
4. generate_fast() vs HF generate() comparison
"""

import time
import torch
from torch.nn import functional as F
from qwen_tts import Qwen3TTSModel

torch.set_float32_matmul_precision('high')

NUM_WARMUP = 3
NUM_TIMED = 10


def cuda_timed(fn, warmup=NUM_WARMUP, repeats=NUM_TIMED):
    """Run fn with CUDA sync timing, return list of durations in seconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def print_stats(label, times_sec):
    ms = [t * 1000 for t in times_sec]
    mean = sum(ms) / len(ms)
    print(f"  {label}:")
    print(f"    Mean: {mean:.2f}ms  Min: {min(ms):.2f}ms  Max: {max(ms):.2f}ms")
    return mean


def profile_generate():
    print("Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Get internal model for profiling
    tts_model = model.model
    talker = tts_model.talker
    code_predictor = talker.code_predictor
    num_codebooks = talker.config.num_code_groups - 1  # 31

    print(f"\nModel structure:")
    print(f"  Talker: {talker.__class__.__name__}")
    print(f"  CodePredictor: {code_predictor.__class__.__name__}")
    print(f"  CodePredictor.model: {code_predictor.model.__class__.__name__}")
    print(f"  Num codebook groups: {talker.config.num_code_groups} (→ {num_codebooks} sequential steps)")
    print(f"  CodePredictor hidden_size: {code_predictor.config.hidden_size}")
    print(f"  CodePredictor num_layers: {code_predictor.config.num_hidden_layers}")
    print(f"  CodePredictor vocab_size: {code_predictor.config.vocab_size}")

    # Check attention implementation
    print(f"\nAttention implementation:")
    print(f"  Talker: {talker.config._attn_implementation}")
    print(f"  CodePredictor: {code_predictor.config._attn_implementation}")

    # Setup dummy inputs for code_predictor
    batch_size = 1
    hidden_size = talker.config.hidden_size
    cp_hidden_size = code_predictor.config.hidden_size
    device = talker.device
    dtype = next(talker.parameters()).dtype

    past_hidden = torch.randn(batch_size, 1, hidden_size, device=device, dtype=dtype)
    last_id_hidden = torch.randn(batch_size, 1, hidden_size, device=device, dtype=dtype)
    inputs_embeds = torch.cat((past_hidden, last_id_hidden), dim=1)

    # =========================================================================
    print("\n" + "=" * 80)
    print(f"BENCHMARK 1: {num_codebooks}x raw model.forward() sequential (prefill + {num_codebooks-1} decode)")
    print("=" * 80)

    projected = code_predictor.small_to_mtp_projection(inputs_embeds)

    def run_31_forwards():
        with torch.no_grad():
            # Prefill (2 tokens)
            out = code_predictor.model(
                inputs_embeds=projected,
                use_cache=True,
                output_hidden_states=False,
            )
            past_kv = out.past_key_values
            # 30 more decode steps
            dummy_embed = torch.randn(1, 1, cp_hidden_size, device=device, dtype=dtype)
            for _ in range(num_codebooks - 1):
                out = code_predictor.model(
                    inputs_embeds=dummy_embed,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=False,
                )
                past_kv = out.past_key_values

    times = cuda_timed(run_31_forwards)
    mean_raw = print_stats(f"{num_codebooks}x raw model.forward()", times)

    # =========================================================================
    print("\n" + "=" * 80)
    print("BENCHMARK 2: HF generate() — output_hidden_states=True vs False")
    print("=" * 80)

    def run_hf_generate_hs_true():
        with torch.no_grad():
            code_predictor.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=num_codebooks,
                do_sample=True,
                top_k=50,
                temperature=1.0,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

    def run_hf_generate_hs_false():
        with torch.no_grad():
            code_predictor.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=num_codebooks,
                do_sample=True,
                top_k=50,
                temperature=1.0,
                output_hidden_states=False,
                return_dict_in_generate=True,
            )

    times_hs_true = cuda_timed(run_hf_generate_hs_true)
    mean_hs_true = print_stats("HF generate(output_hidden_states=True)", times_hs_true)

    times_hs_false = cuda_timed(run_hf_generate_hs_false)
    mean_hs_false = print_stats("HF generate(output_hidden_states=False)", times_hs_false)

    print(f"\n  hidden_states overhead: {mean_hs_true - mean_hs_false:.2f}ms")
    print(f"  HF generate() overhead vs raw forwards: {mean_hs_false - mean_raw:.2f}ms")

    # =========================================================================
    print("\n" + "=" * 80)
    print(f"BENCHMARK 3: Sampling overhead ({num_codebooks}x top_k=50 sampling)")
    print("=" * 80)

    dummy_logits = torch.randn(1, code_predictor.config.vocab_size, device=device, dtype=dtype)

    def run_sampling():
        with torch.no_grad():
            for _ in range(num_codebooks):
                logits = dummy_logits / 1.0  # temperature
                top_k_val = min(50, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k_val)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
                probs = F.softmax(logits, dim=-1)
                torch.multinomial(probs, num_samples=1)

    times_sampling = cuda_timed(run_sampling)
    print_stats(f"{num_codebooks}x sampling (top_k=50)", times_sampling)

    # =========================================================================
    print("\n" + "=" * 80)
    print(f"BENCHMARK 4: Embedding lookups ({num_codebooks}x)")
    print("=" * 80)

    dummy_token = torch.randint(0, code_predictor.config.vocab_size, (1, 1), device=device)

    def run_embeddings():
        with torch.no_grad():
            for i in range(num_codebooks):
                code_predictor.model.codec_embedding[i](dummy_token)

    times_embed = cuda_timed(run_embeddings)
    print_stats(f"{num_codebooks}x codec_embedding lookup", times_embed)

    # =========================================================================
    print("\n" + "=" * 80)
    print(f"BENCHMARK 5: lm_head projections ({num_codebooks}x)")
    print("=" * 80)

    dummy_hidden = torch.randn(1, cp_hidden_size, device=device, dtype=dtype)

    def run_lm_heads():
        with torch.no_grad():
            for i in range(num_codebooks):
                code_predictor.lm_head[i](dummy_hidden)

    times_lm = cuda_timed(run_lm_heads)
    print_stats(f"{num_codebooks}x lm_head projection", times_lm)

    # =========================================================================
    print("\n" + "=" * 80)
    print("BENCHMARK 6: generate_fast() vs HF generate()")
    print("=" * 80)

    def run_generate_fast():
        with torch.no_grad():
            torch.compiler.cudagraph_mark_step_begin()
            code_predictor.generate_fast(
                inputs_embeds=inputs_embeds,
                num_codebooks=num_codebooks,
                do_sample=True,
                temperature=1.0,
                top_k=50,
            )

    times_fast = cuda_timed(run_generate_fast)
    mean_fast = print_stats("generate_fast()", times_fast)
    print(f"\n  Speedup vs HF generate(hs=True):  {mean_hs_true / mean_fast:.2f}x")
    print(f"  Speedup vs HF generate(hs=False): {mean_hs_false / mean_fast:.2f}x")
    print(f"  Speedup vs raw forwards:           {mean_raw / mean_fast:.2f}x")

    # =========================================================================
    print("\n" + "=" * 80)
    print("BENCHMARK 6b: generate_fast() with stacked weights (prepare_fast_weights)")
    print("=" * 80)

    code_predictor.prepare_fast_weights()

    times_fast_stacked = cuda_timed(run_generate_fast)
    mean_fast_stacked = print_stats("generate_fast() + stacked weights", times_fast_stacked)
    print(f"  Speedup vs generate_fast() (no stacked): {mean_fast / mean_fast_stacked:.2f}x")

    # =========================================================================
    print("\n" + "=" * 80)
    print("BENCHMARK 6c: generate_fast() with CUDA graph capture (experimental)")
    print("=" * 80)

    try:
        code_predictor.capture_codebook_cuda_graph(warmup_runs=3)

        def run_generate_fast_cg():
            with torch.no_grad():
                code_predictor.generate_fast(
                    inputs_embeds=inputs_embeds,
                    num_codebooks=num_codebooks,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                )

        times_fast_cg = cuda_timed(run_generate_fast_cg)
        mean_fast_cg = print_stats("generate_fast() + CUDA graph", times_fast_cg)
        print(f"  Speedup vs HF generate(hs=True):     {mean_hs_true / mean_fast_cg:.2f}x")
        print(f"  Speedup vs generate_fast() (stacked): {mean_fast_stacked / mean_fast_cg:.2f}x")

        # Disable for remaining benchmarks
        code_predictor._has_codebook_cuda_graph = False
    except Exception as e:
        print(f"  CUDA graph capture failed: {e}")
        print("  (This is expected if model.forward uses operations incompatible with CUDA graphs)")
        mean_fast_cg = None

    # =========================================================================
    print("\n" + "=" * 80)
    print("BENCHMARK 7: single model.forward() decode step (1 token, with KV cache)")
    print("=" * 80)

    # Prepare a KV cache via prefill
    with torch.no_grad():
        prefill_out = code_predictor.model(
            inputs_embeds=projected,
            use_cache=True,
            output_hidden_states=False,
        )
        warm_past_kv = prefill_out.past_key_values

    single_embed = torch.randn(1, 1, cp_hidden_size, device=device, dtype=dtype)

    def run_single_decode():
        with torch.no_grad():
            code_predictor.model(
                inputs_embeds=single_embed,
                past_key_values=warm_past_kv,
                use_cache=True,
                output_hidden_states=False,
            )

    times_single = cuda_timed(run_single_decode)
    mean_single = print_stats("single decode step", times_single)
    print(f"  Estimated {num_codebooks}x decode: {mean_single * num_codebooks:.2f}ms")

    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  {num_codebooks}x raw model.forward():           {mean_raw:.2f}ms")
    print(f"  HF generate(hs=True):              {mean_hs_true:.2f}ms")
    print(f"  HF generate(hs=False):             {mean_hs_false:.2f}ms")
    print(f"  generate_fast():                   {mean_fast:.2f}ms")
    print(f"  generate_fast() + stacked:         {mean_fast_stacked:.2f}ms")
    if mean_fast_cg is not None:
        print(f"  generate_fast() + CUDA graph:      {mean_fast_cg:.2f}ms")
    print(f"  ---")
    print(f"  hidden_states overhead:             {mean_hs_true - mean_hs_false:.2f}ms")
    print(f"  HF wrapper overhead:               {mean_hs_false - mean_raw:.2f}ms")
    print(f"  generate_fast() vs raw overhead:   {mean_fast - mean_raw:.2f}ms")
    print(f"  Stacked weights improvement:       {mean_fast - mean_fast_stacked:.2f}ms")

    # =========================================================================
    print("\n" + "=" * 80)
    print("TORCH PROFILER: streaming generation (5 chunks)")
    print("=" * 80)

    ref_audio_path = "../neurona-10sec.wav"
    ref_text = "Тестовый текст для профилирования."

    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
    )

    test_text = "Привет, это тест профилирования генерации речи."

    # Warmup
    print("\nWarmup run...")
    for chunk, sr in model.stream_generate_voice_clone(
        text="Раз два три.",
        language="Russian",
        voice_clone_prompt=voice_clone_prompt,
        emit_every_frames=4,
    ):
        pass

    print("Profiling with torch.profiler...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        chunk_count = 0
        for chunk, sr in model.stream_generate_voice_clone(
            text=test_text,
            language="Russian",
            voice_clone_prompt=voice_clone_prompt,
            emit_every_frames=4,
        ):
            chunk_count += 1
            if chunk_count >= 5:
                break

    print("\nTOP 20 CUDA OPERATIONS BY TIME:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("\nTOP 20 CPU OPERATIONS BY TIME:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    print("\nDone profiling!")


if __name__ == "__main__":
    profile_generate()
