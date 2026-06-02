"""Gate C2-L12 — MLA forward."""

import torch
from pathlib import Path

from llmc.data import CharTokenizer, load_text
from llmc.deepseek_v2 import DeepSeekV2, DeepSeekV2Config

DATA = Path(__file__).resolve().parents[2] / "llm-c-from-scratch" / "data" / "tiny_shakespeare.txt"


def test_mla_forward():
    text = load_text(DATA)[:1500] if DATA.is_file() else "ab" * 200
    tok = CharTokenizer.from_text(text)
    cfg = DeepSeekV2Config.tiny(tok.vocab_size, 32)
    model = DeepSeekV2(cfg)
    x = torch.randint(0, tok.vocab_size, (1, 12))
    logits, _ = model(x, x)
    assert logits.shape[-1] == tok.vocab_size
