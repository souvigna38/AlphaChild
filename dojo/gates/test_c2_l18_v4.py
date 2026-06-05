"""Gate C2-L18 — V4 hash-MoE / SwiGLU imports."""

import torch

from llmc.deepseek_v4 import SwiGLUExpert, HashMoE, DeepSeekV4Config


def test_v4_modules():
    cfg = DeepSeekV4Config.tiny(vocab_size=64)
    expert = SwiGLUExpert(cfg)
    x = torch.randn(2, 8, cfg.hidden_size)
    y = expert(x)
    assert y.shape == x.shape
    moe = HashMoE(cfg)
    ids = torch.zeros(2, 8, dtype=torch.long)
    y2 = moe(x, ids)
    assert y2.shape == x.shape
