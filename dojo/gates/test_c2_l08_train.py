"""Gate C2-L08 — short training step."""

import torch
from pathlib import Path

from llmc.data import CharTokenizer, load_text, train_val_split
from llmc.model import GPT, GPTConfig
from llmc.train import Trainer, TrainConfig

DATA = Path(__file__).resolve().parents[2] / "llm-c-from-scratch" / "data" / "tiny_shakespeare.txt"


def test_short_train():
    text = load_text(DATA)[:2000] if DATA.is_file() else "a" * 500
    train_text, val_text = train_val_split(text)
    tok = CharTokenizer.from_text(text)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    model = GPT(GPTConfig.tiny(tok.vocab_size, 32))
    trainer = Trainer(
        model,
        train_ids,
        val_ids,
        TrainConfig(max_steps=3, batch_size=4, eval_interval=3, eval_iters=1),
    )
    trainer.train()
