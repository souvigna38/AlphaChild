"""Gate C2-L15 — DeepSeek-V2 micro-train."""

import torch
from pathlib import Path

from llmc.data import CharTokenizer, load_text, train_val_split
from llmc.deepseek_v2 import DeepSeekV2, DeepSeekV2Config
from llmc.train import Trainer, TrainConfig

DATA = Path(__file__).resolve().parents[2] / "llm-c-from-scratch" / "data" / "tiny_shakespeare.txt"


def test_train_v2_short():
    text = load_text(DATA)[:2500] if DATA.is_file() else "a" * 800
    train_text, val_text = train_val_split(text)
    tok = CharTokenizer.from_text(text)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    cfg = DeepSeekV2Config.tiny(tok.vocab_size, 32)
    model = DeepSeekV2(cfg)
    trainer = Trainer(
        model,
        train_ids,
        val_ids,
        TrainConfig(max_steps=3, batch_size=2, eval_interval=3, eval_iters=1),
    )
    trainer.train()
