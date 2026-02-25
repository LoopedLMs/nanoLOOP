"""Procedurally generated reasoning tasks for training looped transformers.

Each task is a function: (rng: np.random.Generator, level: int) -> str
that returns a string like "12+34=46" where "=" separates question from answer.
The model is trained to predict only the answer tokens (after "=").

Task spec format for multi-task training:
    "addition:1-5 multiplication:1-3 sat:2-4"
Each entry is task_name:min_level-max_level (or task_name:level for fixed level).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from tasks.arithmetic import addition, addition_rev, multiplication, subtraction
from tasks.graph import graph_path
from tasks.grid import grid_transform
from tasks.maze import maze_path
from tasks.sat import sat

# --- Vocabulary ---------------------------------------------------------
# Shared character-level vocabulary across all reasoning tasks.
# Special tokens: PAD=0, BOS=1, EOS=2. Printable chars start at 3.

PAD = 0
BOS = 1
EOS = 2

CHARS = "0123456789abcdefghijklmnopqrstuvwxyz+-*~|&()#.,;=\n>"
STOI: dict[str, int] = {ch: i + 3 for i, ch in enumerate(CHARS)}
ITOS: dict[int, str] = {i: ch for ch, i in STOI.items()}
ITOS[PAD] = ""
ITOS[BOS] = ""
ITOS[EOS] = ""

SEP = STOI["="]

VOCAB_SIZE = 64  # padded for GPU efficiency (55 used)


def encode(s: str) -> list[int]:
    """Encode a string to a list of token ids."""
    return [STOI[ch] for ch in s]


def decode(tokens: list[int]) -> str:
    """Decode a list of token ids to a string."""
    return "".join(ITOS.get(t, "?") for t in tokens)


# --- Task Registry ------------------------------------------------------

# Type alias for task generator functions
type TaskFn = Callable[[np.random.Generator, int], str]

TASK_REGISTRY: dict[str, TaskFn] = {
    "addition": addition,
    "addition_rev": addition_rev,
    "subtraction": subtraction,
    "multiplication": multiplication,
    "sat": sat,
    "grid": grid_transform,
    "graph": graph_path,
    "maze": maze_path,
}


# --- TaskMix ------------------------------------------------------------


class TaskMix:
    """Weighted mixture of reasoning tasks at various difficulty levels.

    Spec format: "task:min-max task:min-max ..."
    Examples:
        "addition:1-5"
        "addition:1-5 multiplication:1-3 sat:2-4"
        "maze:3"  (fixed level 3)
    """

    def __init__(self, entries: list[tuple[TaskFn, int, int]]) -> None:
        self.entries = entries
        self.rng = np.random.default_rng()

    @staticmethod
    def from_spec(spec: str) -> TaskMix:
        """Parse a task spec string into a TaskMix."""
        entries: list[tuple[TaskFn, int, int]] = []
        for token in spec.split():
            name, levels = token.split(":")
            if name not in TASK_REGISTRY:
                raise ValueError(f"Unknown task: {name!r}. Available: {list(TASK_REGISTRY)}")
            if "-" in levels:
                lo, hi = levels.split("-")
                entries.append((TASK_REGISTRY[name], int(lo), int(hi)))
            else:
                lvl = int(levels)
                entries.append((TASK_REGISTRY[name], lvl, lvl))
        return TaskMix(entries)

    def generate_one(self, block_size: int) -> tuple[list[int], int]:
        """Generate one encoded example that fits in block_size.

        Returns (full_tokens, sep_idx) where full_tokens includes BOS and EOS,
        and sep_idx is the position of "=" in full_tokens.
        """
        for _ in range(20):  # retry if too long
            task_fn, lo, hi = self.entries[int(self.rng.integers(len(self.entries)))]
            level = int(self.rng.integers(lo, hi + 1))
            text = task_fn(self.rng, level)
            tokens = encode(text)
            full = [BOS] + tokens + [EOS]
            if len(full) <= block_size + 1:
                sep_idx = full.index(SEP)
                return full, sep_idx
        # last resort: return truncated
        full = full[: block_size + 1]
        sep_idx = min(full.index(SEP) if SEP in full else len(full) - 2, len(full) - 2)
        return full, sep_idx

    def get_batch(self, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of task examples.

        Returns (x, y) where:
            x: (batch_size, block_size) input token ids
            y: (batch_size, block_size) target ids, -1 for masked positions
        """
        x = torch.full((batch_size, block_size), PAD, dtype=torch.long)
        y = torch.full((batch_size, block_size), -1, dtype=torch.long)

        for b in range(batch_size):
            full, sep_idx = self.generate_one(block_size)
            seq_len = min(len(full) - 1, block_size)

            x[b, :seq_len] = torch.tensor(full[:seq_len])
            targets = torch.tensor(full[1 : seq_len + 1])
            # Mask question tokens: everything up to (not including) the first
            # answer token prediction. sep_idx is where "=" sits in full;
            # in the target tensor, position sep_idx predicts the first answer
            # token, so we mask positions 0..sep_idx-1.
            targets[: min(sep_idx, seq_len)] = -1
            y[b, :seq_len] = targets

        return x, y
