from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    # Optional debug
    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:
     # Requirements:
    # - Let N = batch.input_ids.shape[0] be the number of sampled completions.
    # - If shuffle=True, permute indices with torch.randperm using the provided generator.
    # - Otherwise iterate in the original order 0, 1, ..., N-1.
    # - Slice ALL tensor fields consistently with the same minibatch indices.
    # - Keep task_names / completion_texts aligned with the same indices when present.
    # - If device is not None, move the minibatch to that device before yielding.
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")

    n = batch.input_ids.shape[0]
    indices = torch.randperm(n, generator=generator) if shuffle else None
    for start in range(0, n, minibatch_size):
        stop = min(start + minibatch_size, n)
        mb_indices = indices[start:stop] if shuffle else slice(start, stop)
        debug_indices = (
            mb_indices.tolist() if shuffle else range(start, stop)
        ) if batch.task_names is not None or batch.completion_texts is not None else ()
        minibatch = RolloutBatch(
            input_ids=batch.input_ids[mb_indices],
            attention_mask=batch.attention_mask[mb_indices],
            completion_mask=batch.completion_mask[mb_indices],
            old_logprobs=batch.old_logprobs[mb_indices],
            ref_logprobs=batch.ref_logprobs[mb_indices],
            rewards=batch.rewards[mb_indices],
            advantages=batch.advantages[mb_indices],
            task_names=(
                [batch.task_names[i] for i in debug_indices]
                if batch.task_names is not None else None
            ),
            completion_texts=(
                [batch.completion_texts[i] for i in debug_indices]
                if batch.completion_texts is not None else None
            ),
        )
        yield minibatch.to(device) if device is not None else minibatch
