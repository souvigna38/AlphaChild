"""Gate C2-L04 — causal attention shapes."""

import torch

from llmc.model import CausalSelfAttention, GPTConfig


def test_causal_attention():
    cfg = GPTConfig(vocab_size=64, n_embd=32, n_head=4, n_layer=1, block_size=16)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 8, cfg.n_embd)
    y = attn(x)
    assert y.shape == x.shape
