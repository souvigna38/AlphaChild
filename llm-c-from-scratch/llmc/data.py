"""Text loading and character-level tokenization (llm.c uses GPT-2 BPE; we start simpler)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def train_val_split(data: str, train_frac: float = 0.9) -> tuple[str, str]:
    split = int(train_frac * len(data))
    return data[:split], data[split:]


@dataclass
class CharTokenizer:
    """Map characters to integers — same idea as early nanoGPT before BPE."""

    chars: list[str]

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        unique = sorted(set(text))
        return cls(chars=unique)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        stoi = {ch: i for i, ch in enumerate(self.chars)}
        return [stoi[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        itos = {i: ch for i, ch in enumerate(self.chars)}
        return "".join(itos[i] for i in tokens)

    def encode_tensor(self, text: str, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.tensor(self.encode(text), dtype=torch.long, device=device)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random (B, T) inputs and (B, T) next-token targets."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
