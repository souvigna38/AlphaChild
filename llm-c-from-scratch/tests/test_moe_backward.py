"""DeepSeekMoE backward — PyTorch autograd reference."""

import torch
import torch.nn.functional as F

from llmc.deepseek_v2 import DeepSeekV2Config, DeepSeekMoE


def test_moe_backward_autograd():
    cfg = DeepSeekV2Config.tiny(vocab_size=32, block_size=8)
    moe = DeepSeekMoE(cfg)
    x = torch.randn(2, 6, cfg.n_embd, requires_grad=True)
    y = moe(x)
    y.sum().backward()
    assert x.grad is not None and x.grad.abs().mean() > 0
    assert moe.gate.weight.grad is not None
    assert any(e.w1.weight.grad is not None for e in moe.experts)


def test_moe_router_topk():
    cfg = DeepSeekV2Config.tiny(32, 8)
    moe = DeepSeekMoE(cfg)
    x = torch.randn(1, 1, cfg.n_embd)
    logits = moe.gate(x.view(-1, cfg.n_embd))
    probs = F.softmax(logits, dim=-1)
    topw, topi = torch.topk(probs, cfg.num_experts_per_tok, dim=-1)
    assert topi.shape[-1] == cfg.num_experts_per_tok
