"""MLA backward: PyTorch autograd reference (C backward tested via make test_v2)."""

import torch

from llmc.deepseek_v2 import DeepSeekV2Config, MultiHeadLatentAttention


def test_mla_backward_matches_autograd():
    cfg = DeepSeekV2Config.tiny(vocab_size=32, block_size=16)
    attn = MultiHeadLatentAttention(cfg)
    x = torch.randn(1, 12, cfg.n_embd, requires_grad=True)
    y = attn(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert attn.wq.weight.grad is not None
    assert attn.w_dkv.weight.grad is not None
    assert float(x.grad.abs().mean()) > 0.0
