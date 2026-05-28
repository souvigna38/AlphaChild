"""Training loop — mirrors train_gpt2.py / llm.c train loop at a small scale."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import AdamW

from llmc.data import get_batch
from llmc.model import GPT


@dataclass
class TrainConfig:
    max_steps: int = 500
    batch_size: int = 32
    learning_rate: float = 3e-4
    eval_interval: int = 100
    eval_iters: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0


class Trainer:
    def __init__(
        self,
        model: GPT,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        config: TrainConfig,
        device: str | torch.device = "cpu",
    ):
        self.model = model.to(device)
        self.train_data = train_data.to(device)
        self.val_data = val_data.to(device)
        self.config = config
        self.device = device
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    @torch.no_grad()
    def estimate_loss(self) -> tuple[float, float]:
        out: dict[str, float] = {}
        self.model.eval()
        for split, data in [("train", self.train_data), ("val", self.val_data)]:
            losses = []
            for _ in range(self.config.eval_iters):
                x, y = get_batch(data, self.config.batch_size, self.model.config.block_size, self.device)
                _, loss = self.model(x, y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        self.model.train()
        return out["train"], out["val"]

    def train_step(self) -> float:
        x, y = get_batch(
            self.train_data,
            self.config.batch_size,
            self.model.config.block_size,
            self.device,
        )
        _, loss = self.model(x, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return loss.item()

    def train(self) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for step in range(self.config.max_steps):
            loss = self.train_step()
            if step % self.config.eval_interval == 0 or step == self.config.max_steps - 1:
                train_loss, val_loss = self.estimate_loss()
                history.append({"step": step, "train": train_loss, "val": val_loss, "batch": loss})
        return history
