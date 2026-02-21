"""PagedEngine — central orchestrator for paged attention TTS inference.

Ties together paged model wrappers, model runners, schedulers, and block managers.
Provides prefill() / step() / cleanup() methods that replace the HF talker.forward()
+ code_predictor.generate() inner loop in stream_generate_pcm().

The boundary is codec_ids [B, 16] per frame — everything before (input preparation)
and after (buffering, emit logic, speech tokenizer decode, crossfade) stays identical.
"""

import logging
import math
from typing import Optional

import torch
from torch import nn

from .models.paged_talker import PagedTalker
from .models.paged_predictor import PagedPredictor
from .model_runner import ModelRunner
from .scheduler import TalkerScheduler, PredictorScheduler
from .sequence import Sequence, SequenceStatus
from .context import set_context, reset_context
from .sampler import Sampler

logger = logging.getLogger(__name__)


class PagedEngine:
    """Paged attention engine for Qwen3-TTS inference.

    Wraps the HF model's Talker and CodePredictor with paged attention,
    sharing all weight tensors by reference. Only allocates new KV cache
    tensors and CUDA graph buffers.
    """

    def __init__(
        self,
        hf_model: nn.Module,
        gpu_memory_utilization: float = 0.3,
        enforce_eager: bool = False,
        max_num_seqs: int = 32,
        max_model_len: int = 4096,
        block_size: int = 256,
    ):
        """Initialize paged engine from an HF Qwen3TTSForConditionalGeneration model.

        Args:
            hf_model: The HF Qwen3TTSForConditionalGeneration instance
            gpu_memory_utilization: Fraction of GPU memory for KV cache
            enforce_eager: Skip CUDA graph capture (debug mode)
            max_num_seqs: Max concurrent sequences for continuous batching
            max_model_len: Max sequence length
            block_size: KV cache block size (tokens per block)
        """
        self.hf_model = hf_model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.block_size = block_size

        talker_config = hf_model.config.talker_config
        predictor_config = talker_config.code_predictor_config

        # Build paged wrappers (weight sharing, no copies)
        self.paged_talker = PagedTalker(hf_model.talker, talker_config)
        self.paged_predictor = PagedPredictor(
            hf_model.talker.code_predictor, predictor_config, talker_config
        )

        # Model runners (handle KV cache, input prep, CUDA graphs)
        self.talker_runner = ModelRunner(
            model=self.paged_talker,
            model_config=talker_config,
            block_size=block_size,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
        )
        self.predictor_runner = ModelRunner(
            model=self.paged_predictor,
            model_config=predictor_config,
            block_size=block_size,
            max_num_seqs=max_num_seqs,
            max_model_len=64,  # Predictor only needs 17 tokens per frame
            enforce_eager=enforce_eager,
            input_hidden_size=talker_config.hidden_size,
        )

        # Samplers
        self.talker_sampler = Sampler()
        self.predictor_sampler = Sampler()

        # Store configs for reference
        self.talker_config = talker_config
        self.predictor_config = predictor_config
        self.num_code_groups = talker_config.num_code_groups  # 16

        self._warmed_up = False

    def warmup(self) -> None:
        """Allocate KV caches and capture CUDA graphs.

        Must run in the same thread context as inference (CUDA graph TLS requirement).
        """
        if self._warmed_up:
            return

        logger.info("[PagedEngine] Warming up talker model runner...")
        self.talker_runner.warmup_model()
        talker_blocks = self.talker_runner.allocate_kv_cache(self.gpu_memory_utilization)
        logger.info(f"[PagedEngine] Talker KV cache: {talker_blocks} blocks allocated")

        logger.info("[PagedEngine] Warming up predictor model runner...")
        self.predictor_runner.warmup_model()
        # Predictor needs ceil(max_model_len / block_size) blocks per sequence.
        # Allocate exactly what's needed — VRAM-based calculation fails here because
        # the talker's KV cache already dominates used memory, making the budget negative.
        predictor_blocks_needed = math.ceil(64 / self.block_size) * self.max_num_seqs
        predictor_blocks = self.predictor_runner.allocate_kv_cache(num_blocks=predictor_blocks_needed)
        logger.info(f"[PagedEngine] Predictor KV cache: {predictor_blocks} blocks allocated")

        if not self.enforce_eager:
            logger.info("[PagedEngine] Capturing CUDA graphs...")
            self.talker_runner.capture_cudagraph()
            self.predictor_runner.capture_cudagraph()
            logger.info("[PagedEngine] CUDA graphs captured")

        # Initialize schedulers (need num_blocks from allocation)
        self.talker_scheduler = TalkerScheduler(
            num_kvcache_blocks=talker_blocks,
            block_size=self.block_size,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_model_len * self.max_num_seqs,
        )
        self.predictor_scheduler = PredictorScheduler(
            num_kvcache_blocks=predictor_blocks,
            block_size=self.block_size,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=64 * self.max_num_seqs,
        )

        self._warmed_up = True
        logger.info("[PagedEngine] Warmup complete")

    @torch.inference_mode()
    def prefill(
        self,
        talker_input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.9,
        top_k: int = 50,
        request_id: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Sequence]:
        """Run talker prefill phase.

        Args:
            talker_input_embeds: [B, seq_len, hidden_size] from _build_talker_inputs()
            attention_mask: [B, seq_len] padding mask (used to compute cu_seqlens)
            temperature: Sampling temperature
            top_k: Top-k for sampling
            request_id: Optional request identifier for continuous batching

        Returns:
            (first_token, last_hidden_state, sequence_handle)
        """
        assert self._warmed_up, "Call warmup() before prefill()"

        # Create sequence from input embeddings
        # Convert attention_mask to actual sequence lengths for cu_seqlens
        if attention_mask is not None:
            seq_len = attention_mask.sum(dim=-1).item()
        else:
            seq_len = talker_input_embeds.shape[1]

        # Build list of per-token embeddings for the sequence
        embeds_list = [talker_input_embeds[0, i:i+1, :] for i in range(int(seq_len))]

        seq = Sequence(
            token_ids=[0] * int(seq_len),  # Placeholder IDs for block allocation
            temperature=temperature,
            max_tokens=self.max_model_len,
            ignore_eos=True,
            input_embeds=talker_input_embeds[0, :int(seq_len), :],  # [seq_len, hidden]
            request_id=request_id,
        )

        # Add to scheduler and run prefill
        self.talker_scheduler.add(seq)
        if request_id:
            self.talker_scheduler.request_id_to_seq[request_id] = seq

        seqs, is_prefill = self.talker_scheduler.schedule()
        assert is_prefill and seqs

        hidden_states, logits = self.talker_runner.run(seqs, is_prefill=True)

        # Sample first token
        temperatures = torch.tensor([temperature], dtype=torch.float32, device=logits.device)
        first_token = self.talker_sampler(logits, temperatures, top_k=top_k)

        # Postprocess: record token in sequence
        self.talker_scheduler.postprocess(
            seqs, first_token.tolist(),
            [hidden_states[i] for i in range(len(seqs))],
        )

        return first_token, hidden_states[0], seq

    @torch.inference_mode()
    def step(
        self,
        seq: Sequence,
        input_embeds: torch.Tensor,
        temperature: float = 0.9,
        top_k: int = 50,
        subtalker_temperature: float = 0.9,
        subtalker_top_k: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one Talker decode step + CodePredictor generation.

        Args:
            seq: Sequence handle from prefill()
            input_embeds: [1, hidden_size] next input embedding (sum of codec embeddings
                         + text conditioning)
            temperature: Talker sampling temperature
            top_k: Talker top-k
            subtalker_temperature: CodePredictor sampling temperature
            subtalker_top_k: CodePredictor top-k

        Returns:
            (next_talker_token, codec_ids [1, 16], talker_logits)
        """
        # 1. Talker decode: set input_embeds and run single-token step
        seq.decode_input_embeds = input_embeds

        seqs, is_prefill = self.talker_scheduler.schedule()
        if not seqs:
            raise RuntimeError("TalkerScheduler returned no sequences for decode")

        hidden_states, talker_logits = self.talker_runner.run(seqs, is_prefill=False)
        talker_hidden = hidden_states[0]  # [hidden_size]

        # Sample talker token (first codebook)
        temperatures = torch.tensor([temperature], dtype=torch.float32, device=talker_logits.device)
        talker_token = self.talker_sampler(talker_logits, temperatures, top_k=top_k)

        # Update talker sequence
        self.talker_scheduler.postprocess(
            seqs, talker_token.tolist(),
            [talker_hidden for _ in seqs],
        )

        # 2. CodePredictor: generate 15 codebook tokens
        codec_ids = self._generate_codebooks(
            talker_hidden=talker_hidden,
            first_token=talker_token,
            temperature=subtalker_temperature,
            top_k=subtalker_top_k,
        )

        return talker_token, codec_ids, talker_logits

    @torch.inference_mode()
    def _generate_codebooks(
        self,
        talker_hidden: torch.Tensor,
        first_token: torch.Tensor,
        temperature: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generate 15 codebook tokens using the CodePredictor.

        The predictor runs a short sequence per frame:
        - Position 0: talker_hidden_state (projected to predictor dim)
        - Position 1: first codec token embedding
        - Positions 2-16: auto-regressive decode for codebooks 1-15

        Returns:
            codec_ids: [1, 16] tensor with all 16 codebook tokens
        """
        num_sub_codebooks = self.num_code_groups - 1  # 15

        # Build predictor prefill input: [talker_hidden, first_codec_embed]
        first_codec_embed = self.paged_talker.get_input_embeddings()(first_token)  # [1, talker_hidden]
        prefill_embeds = torch.cat([
            talker_hidden.unsqueeze(0),  # [1, talker_hidden]
            first_codec_embed,  # [1, talker_hidden]
        ], dim=0).unsqueeze(0)  # [1, 2, talker_hidden]

        # Create short-lived predictor sequence
        pred_seq = Sequence(
            token_ids=[0, 0],  # 2 prefill tokens
            temperature=temperature,
            max_tokens=num_sub_codebooks + 2,
            ignore_eos=True,
            input_embeds=prefill_embeds[0],  # [2, hidden]
        )

        # Prefill
        self.predictor_scheduler.add(pred_seq)
        seqs, is_prefill = self.predictor_scheduler.schedule()
        assert is_prefill

        # Run predictor prefill
        pred_input_embeds, positions = self.predictor_runner.prepare_prefill(seqs)
        hidden_states = self.predictor_runner.run_model(positions, True, pred_input_embeds)

        # Sample first sub-codebook token
        gen_steps = [0]  # codebook index 0 (lm_head[0])
        logits = self.paged_predictor.compute_logits(hidden_states, gen_steps)
        temperatures = torch.tensor([temperature], dtype=torch.float32, device=logits.device)
        token = self.predictor_sampler(logits, temperatures, top_k=top_k)
        reset_context()

        # Record in predictor sequence
        pred_seq.append_token(token.item())
        self.predictor_scheduler.block_manager.may_append(pred_seq)

        all_tokens = [first_token.item(), token.item()]

        # Auto-regressive decode for remaining codebooks
        for step in range(1, num_sub_codebooks):
            # Get codec embedding for the token we just generated
            codec_embed = self.paged_predictor.get_codec_embedding()[step - 1](token)  # [1, talker_hidden]

            # Prepare decode
            pred_seq.decode_input_embeds = codec_embed
            positions_t = torch.tensor([len(pred_seq)], dtype=torch.int64, device="cuda")
            slot = pred_seq.block_table[-1] * self.block_size + pred_seq.last_block_num_tokens - 1
            slot_mapping = torch.tensor([slot], dtype=torch.int32, device="cuda")
            context_lens = torch.tensor([len(pred_seq)], dtype=torch.int32, device="cuda")
            block_tables = self.predictor_runner.prepare_block_tables([pred_seq])

            set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)

            # Run predictor (forward() handles small_to_mtp_projection internally)
            hidden_states = self.predictor_runner.run_model(positions_t, False, codec_embed)

            gen_steps = [step]
            logits = self.paged_predictor.compute_logits(hidden_states, gen_steps)
            token = self.predictor_sampler(logits, temperatures, top_k=top_k)
            reset_context()

            pred_seq.append_token(token.item())
            self.predictor_scheduler.block_manager.may_append(pred_seq)
            all_tokens.append(token.item())

        # Cleanup predictor sequence
        self.predictor_scheduler.block_manager.deallocate(pred_seq)

        # Build codec_ids [1, 16]
        codec_ids = torch.tensor([all_tokens], dtype=torch.long, device="cuda")
        return codec_ids

    def cleanup(self, seq: Sequence) -> None:
        """Deallocate KV blocks for a finished sequence."""
        request_id = seq.request_id
        if request_id:
            self.talker_scheduler.clear_request(request_id)
        else:
            with self.talker_scheduler._lock:
                if seq in self.talker_scheduler.running:
                    self.talker_scheduler.running.remove(seq)
                self.talker_scheduler.block_manager.deallocate(seq)

    def build_next_input_embeds(
        self,
        codec_ids: torch.Tensor,
        trailing_text_hiddens: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        generation_step: int,
    ) -> torch.Tensor:
        """Build the next talker input embedding from codec_ids + text conditioning.

        This replicates the logic from HF's Talker.forward() that builds input_embeds
        for the next AR step from the generated codec_ids and text conditioning.

        Args:
            codec_ids: [1, 16] generated codec tokens
            trailing_text_hiddens: [1, hidden_size] text conditioning
            tts_pad_embed: [1, hidden_size] TTS padding embedding
            generation_step: Current generation step index

        Returns:
            input_embeds: [1, hidden_size] for next talker decode step
        """
        # Sum codec embeddings: first codebook via talker's codec_embedding,
        # remaining via predictor's codec_embeddings
        talker_embed = self.paged_talker.get_input_embeddings()
        predictor_embeds = self.paged_predictor.get_codec_embedding()

        # First codebook
        embed_sum = talker_embed(codec_ids[:, 0:1]).squeeze(1)  # [1, hidden]

        # Remaining codebooks
        for i in range(self.num_code_groups - 1):
            embed_sum = embed_sum + predictor_embeds[i](codec_ids[:, i + 1:i + 2]).squeeze(1)

        # Add text conditioning
        embed_sum = embed_sum + trailing_text_hiddens + tts_pad_embed

        return embed_sum.unsqueeze(1)  # [1, 1, hidden] for decode
