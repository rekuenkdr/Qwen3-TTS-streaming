"""Talker and Predictor schedulers for continuous batching.

TalkerScheduler: persistent sequences (via request_id_to_seq), only schedules
decode when decode_input_embeds is set (interface has fed next input).

PredictorScheduler: simpler, short-lived sequences per frame.

Both include threading.Lock for thread safety with concurrent API requests.
"""

import threading
from collections import deque
from typing import Optional

from .sequence import Sequence, SequenceStatus
from .block_manager import BlockManager


class Scheduler:
    """Base scheduler with prefill/decode scheduling and block management."""

    def __init__(
        self,
        num_kvcache_blocks: int,
        block_size: int = 256,
        max_num_seqs: int = 32,
        max_num_batched_tokens: int = 16384,
        eos: int = -1,
    ):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos = eos
        self.block_manager = BlockManager(num_kvcache_blocks, block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self._lock = threading.Lock()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        with self._lock:
            self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        with self._lock:
            return self._schedule_locked()

    def _schedule_locked(self) -> tuple[list[Sequence], bool]:
        # Prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if (
                num_batched_tokens + len(seq) > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)
            ):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # Decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        if not scheduled_seqs:
            return [], False
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        with self._lock:
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id, last_hidden_state=None)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)


class TalkerScheduler(Scheduler):
    """Scheduler for the Talker model with persistent sequence tracking.

    Only schedules decode when decode_input_embeds is set (the engine has
    computed the next input embedding from codec_ids + text conditioning).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.request_id_to_seq: dict[str, Sequence] = {}

    def _schedule_locked(self) -> tuple[list[Sequence], bool]:
        # Prefill: same as base
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if (
                num_batched_tokens + len(seq) > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)
            ):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # Decode: only schedule seqs with decode_input_embeds set
        run_count = len(self.running)
        for _ in range(run_count):
            if not self.running or num_seqs >= self.max_num_seqs:
                break
            seq = self.running.popleft()
            if len(seq) > 0 and seq.decode_input_embeds is None:
                self.running.append(seq)
                continue
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        if not scheduled_seqs:
            return [], False
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        hidden_states: list[Optional["torch.Tensor"]],
    ) -> None:
        with self._lock:
            for seq, token_id, hs in zip(seqs, token_ids, hidden_states):
                seq.append_token(token_id, last_hidden_state=hs)
                seq.decode_input_embeds = None  # Reset until engine sets next

                if seq.request_id is not None:
                    finish = not seq.ignore_eos and token_id == self.eos
                else:
                    finish = (
                        (not seq.ignore_eos and token_id == self.eos)
                        or seq.num_completion_tokens >= seq.max_tokens
                    )
                if finish:
                    seq.status = SequenceStatus.FINISHED
                    if seq.request_id is not None:
                        self.request_id_to_seq.pop(seq.request_id, None)
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)

    def clear_request(self, request_id: str) -> None:
        with self._lock:
            if request_id in self.request_id_to_seq:
                seq = self.request_id_to_seq.pop(request_id)
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)


class PredictorScheduler(Scheduler):
    """Scheduler for the CodePredictor — simpler, short-lived sequences per frame."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.request_id_to_seq: dict[str, Sequence] = {}

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        with self._lock:
            super().postprocess(seqs, token_ids)
            for seq in seqs:
                if seq.is_finished and seq.request_id is not None:
                    self.request_id_to_seq.pop(seq.request_id, None)

    def clear_request(self, request_id: str) -> None:
        with self._lock:
            if request_id in self.request_id_to_seq:
                seq = self.request_id_to_seq.pop(request_id)
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
