"""Sequence state tracking for paged attention engine.

Each Sequence tracks token IDs, paged KV block table, input embeddings,
and hidden state for continuous batching.
"""

from copy import copy
from enum import Enum, auto
from itertools import count
from typing import Optional

import torch


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: Optional[list[int]],
        temperature: float = 0.9,
        max_tokens: int = 10000,
        ignore_eos: bool = True,
        input_embeds: Optional[torch.Tensor] = None,
        request_id: Optional[str] = None,
    ):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.input_embeds = input_embeds
        self.request_id = request_id
        self.decode_input_embeds: Optional[torch.Tensor] = None
        self.token_ids = copy(token_ids) if token_ids else []
        self.last_token = token_ids[-1] if token_ids else None
        self.num_tokens = len(token_ids) if token_ids else (input_embeds.shape[1] if input_embeds is not None else 0)
        self.num_prompt_tokens = len(token_ids) if token_ids else 0
        self.num_cached_tokens = 0
        self.block_table: list[int] = []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.generation_steps = 0
        self.last_hidden_state: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self) -> int:
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i: int) -> list[int]:
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size:(i + 1) * self.block_size]

    def append_token(self, token_id: int, last_hidden_state: Optional[torch.Tensor] = None) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.last_hidden_state = last_hidden_state
