"""Model runner: KV cache allocation, prefill/decode preparation, CUDA graph capture.

Manages the paged KV cache tensors and CUDA graph capture for both Talker and
Predictor models. Provides prepare_prefill/prepare_decode methods that translate
sequence state into flash_attn-compatible inputs.
"""

from typing import Optional

import torch

from .context import set_context, get_context, reset_context
from .sequence import Sequence
from .sampler import Sampler


class ModelRunner:
    """Handles KV cache allocation, input preparation, and CUDA graph capture."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_config,
        block_size: int = 256,
        max_num_seqs: int = 32,
        max_model_len: int = 4096,
        enforce_eager: bool = False,
        input_hidden_size: Optional[int] = None,
    ):
        self.model = model
        self.model_config = model_config
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self.input_hidden_size = input_hidden_size or model_config.hidden_size
        self.sampler = Sampler()

        # Populated by allocate_kv_cache / capture_cudagraph
        self.kv_cache: Optional[torch.Tensor] = None
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_bs: list[int] = []
        self.graph_pool: Optional[object] = None
        self.graph_vars: dict[str, torch.Tensor] = {}

    @property
    def num_layers(self) -> int:
        return self.model_config.num_hidden_layers

    @property
    def num_kv_heads(self) -> int:
        return self.model_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        hd = getattr(self.model_config, "head_dim", None)
        if hd:
            return hd
        return self.model_config.hidden_size // self.model_config.num_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.model_config.hidden_size

    def warmup_model(self) -> None:
        """Run a dummy forward pass to trigger CUDA initialization and memory stats."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        num_seqs = min(2, self.max_num_seqs)
        dummy_seqs = [
            Sequence([], input_embeds=torch.zeros(1, 8, self.input_hidden_size, device="cuda", dtype=torch.bfloat16))
            for _ in range(num_seqs)
        ]
        # Run forward pass only (no logits) — warmup just needs CUDA init + peak memory stats.
        # Skips self.run() because compute_logits() has model-specific signatures.
        input_embeds, positions = self.prepare_prefill(dummy_seqs)
        self.run_model(positions, True, input_embeds)
        reset_context()
        torch.cuda.empty_cache()

    def allocate_kv_cache(self, gpu_memory_utilization: float = 0.3, num_blocks: int | None = None) -> int:
        """Allocate paged KV cache based on available VRAM or an explicit block count.

        Args:
            gpu_memory_utilization: Fraction of GPU memory for KV cache (used when num_blocks is None).
            num_blocks: If provided, allocate exactly this many blocks (skips VRAM calculation).
                        Useful when the model shares GPU with other components whose memory
                        makes the VRAM-based formula unreliable.

        Returns:
            Number of KV blocks allocated.
        """
        torch_dtype = torch.bfloat16

        block_bytes = (
            2  # K + V
            * self.num_layers
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * torch_dtype.itemsize
        )

        if num_blocks is None:
            free, total = torch.cuda.mem_get_info()
            used = total - free
            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

            num_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
            assert num_blocks > 0, (
                f"Cannot allocate KV cache: need at least 1 block ({block_bytes} bytes), "
                f"but only {total * gpu_memory_utilization - used - peak + current:.0f} bytes available"
            )

        self.kv_cache = torch.empty(
            2, self.num_layers, num_blocks, self.block_size,
            self.num_kv_heads, self.head_dim,
            dtype=torch_dtype, device="cuda",
        )

        # Assign cache slices to each attention layer
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

        return num_blocks

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        return torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def prepare_prefill(self, seqs: list[Sequence]):
        """Prepare inputs for variable-length prefill via flash_attn_varlen_func."""
        positions = []
        input_embeds_list = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            seqlen = len(seq)
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            if seq.input_embeds is not None:
                input_embeds_list.extend(seq.input_embeds[seq.num_cached_tokens:])

            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        input_embeds = torch.cat(
            [e if e.dim() > 1 else e.unsqueeze(0) for e in input_embeds_list], dim=0
        ).to(dtype=torch.bfloat16)
        if input_embeds.device.type != "cuda":
            input_embeds = input_embeds.pin_memory().cuda(non_blocking=True)

        positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(
            True, cu_seqlens_q_t, cu_seqlens_k_t,
            max_seqlen_q, max_seqlen_k,
            slot_mapping_t, None, block_tables,
        )
        return input_embeds, positions_t

    def prepare_decode(self, seqs: list[Sequence]):
        """Prepare inputs for single-token decode step."""
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens_t = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping_t, context_lens=context_lens_t, block_tables=block_tables)
        return positions_t

    def prepare_decode_with_embeds(self, seqs: list[Sequence]):
        """Prepare decode inputs when sequences use input_embeds instead of token IDs (Talker)."""
        positions = []
        slot_mapping = []
        context_lens = []
        input_embeds_list = []
        for seq in seqs:
            emb = seq.decode_input_embeds
            if emb is None:
                raise ValueError(f"Sequence {seq.seq_id} has no decode_input_embeds set")
            input_embeds_list.append(emb)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_embeds = torch.cat(
            [e.reshape(-1, e.shape[-1]) if e.dim() > 1 else e.unsqueeze(0) for e in input_embeds_list],
            dim=0,
        ).to(dtype=torch.bfloat16)
        if input_embeds.device.type != "cuda":
            input_embeds = input_embeds.pin_memory().cuda(non_blocking=True)

        positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens_t = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping_t, context_lens=context_lens_t, block_tables=block_tables)
        return input_embeds, positions_t

    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    @torch.inference_mode()
    def run_model(
        self,
        positions: torch.Tensor,
        is_prefill: bool,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        """Run model forward pass, using CUDA graphs for decode when available."""
        if is_prefill or self.enforce_eager or (input_embeds is not None and input_embeds.size(0) > 512):
            hidden_states = self.model(input_embeds, positions)
        else:
            bs = input_embeds.size(0)
            context = get_context()
            graph = self.graphs.get(next((x for x in self.graph_bs if x >= bs), None))
            if graph is None:
                hidden_states = self.model(input_embeds, positions)
            else:
                gv = self.graph_vars
                gv["input_embeds"][:bs] = input_embeds
                gv["positions"][:bs] = positions
                gv["slot_mapping"].fill_(-1)
                gv["slot_mapping"][:bs] = context.slot_mapping
                gv["context_lens"].zero_()
                gv["context_lens"][:bs] = context.context_lens
                gv["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
                graph.replay()
                hidden_states = gv["outputs"][:bs]

        return hidden_states

    def run(self, seqs: list[Sequence], is_prefill: bool):
        """Run prefill or decode, return (hidden_states, logits)."""
        if is_prefill:
            input_embeds, positions = self.prepare_prefill(seqs)
        else:
            input_embeds, positions = self.prepare_decode_with_embeds(seqs)

        hidden_states = self.run_model(positions, is_prefill, input_embeds)
        logits = self.model.compute_logits(hidden_states)

        if is_prefill:
            # Extract last-token hidden states and logits per sequence
            context = get_context()
            last_indices = context.cu_seqlens_q[1:] - 1
            hidden_states = hidden_states[last_indices].contiguous()
            logits = logits[last_indices].contiguous()

        reset_context()
        return hidden_states, logits

    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        """Capture CUDA graphs for decode at various batch sizes."""
        max_bs = min(self.max_num_seqs, 512)
        max_num_blocks = (self.max_model_len + self.block_size - 1) // self.block_size

        input_embeds = torch.zeros(max_bs, self.input_hidden_size, device="cuda", dtype=torch.bfloat16)
        positions = torch.zeros(max_bs, dtype=torch.int64, device="cuda")
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device="cuda")
        outputs = torch.zeros(max_bs, self.hidden_size, device="cuda", dtype=torch.bfloat16)

        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_embeds[:bs].unsqueeze(1) if input_embeds[:bs].dim() == 1 else input_embeds[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_embeds[:bs], positions[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = {
            "input_embeds": input_embeds,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "outputs": outputs,
        }
