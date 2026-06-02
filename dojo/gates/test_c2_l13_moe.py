"""Gate C2-L13 — MoE block."""

import torch
from pathlib import Path

from llmc.data import CharTokenizer, load_text
from llmc.deepseek_v2 import DeepSeekV2, DeepSeekV2Config

DATA = Path(__file__).resolve().parents[2] / "llm-c-from-scratch" / "data" / "tiny_shakespeare.txt"


def test_moe_block():
    text = load_text(DATA)[:1500] if DATA.is_file() else "ab" * 200
    tok = CharTokenizer.from_text(text)
    cfg = DeepSeekV2Config.tiny(tok.vocab_size, 32)
    model = DeepSeekV2(cfg)
    x = torch.randint(0, tok.vocab_size, (1, 8))
    logits, _ = model(x, x)
    assert len(model.blocks[0].moe.experts) >= 1
