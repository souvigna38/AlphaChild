"""
DeepSeek-V2 (educational, tiny) — PyTorch reference for notebook series 11–16.

Paper: DeepSeek-V2 (MLA + DeepSeekMoE)
https://arxiv.org/html/2405.04434

This is intentionally SMALL and HEAVILY COMMENTED so you can match it to llm.c later.
We simplify decoupled RoPE in MLA (Phase 1). Add partial RoPE in a later notebook/C file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class DeepSeekV2Config:
    """
    Tiny config for laptops. Same *roles* as V2, not the 236B production sizes.
    """

    vocab_size: int
    block_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    # MLA: low-rank KV compression dimension (the thing we cache at inference)
    kv_lora_rank: int = 32
    # MoE
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    num_experts_per_tok: int = 2
    moe_intermediate: int = 192
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6

    @classmethod
    def tiny(cls, vocab_size: int, block_size: int = 64) -> DeepSeekV2Config:
        return cls(vocab_size=vocab_size, block_size=block_size)


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    DeepSeek uses RMSNorm (no mean centering) instead of GPT-2 LayerNorm.
    llm.c: layernorm_forward in train_gpt2.c
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) — core DeepSeek-V2 idea.

    Standard MHA (notebook 4 / llm.c):
      - For each layer we cache FULL keys and values: shape ~(n_head, head_dim) per token.

    MLA:
      - Compress keys+values into ONE small latent vector c_kv per token (dim = kv_lora_rank).
      - At inference, KV cache size ∝ kv_lora_rank instead of n_head * head_dim * 2.

    Educational simplification here:
      - We still compute attention explicitly (like notebook 4).
      - We SHOW the latent cache tensor so you can print its size vs MHA.
    """

    def __init__(self, config: DeepSeekV2Config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.kv_lora_rank = config.kv_lora_rank

        # Query path (could add q_lora in full V2; kept direct for clarity)
        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Down-project hidden state → latent KV (the compressed cache)
        self.w_dkv = nn.Linear(config.n_embd, config.kv_lora_rank, bias=False)
        # Up-project latent → per-head keys and values
        self.w_uk = nn.Linear(config.kv_lora_rank, config.n_embd, bias=False)
        self.w_uv = nn.Linear(config.kv_lora_rank, config.n_embd, bias=False)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        # Expose last latent cache for teaching (B, T, kv_lora_rank)
        self.last_c_kv: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape

        q = self.wq(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        # --- MLA compression path ---
        c_kv = self.w_dkv(x)  # (B, T, kv_lora_rank)  ← this is what V2 prefers to cache
        self.last_c_kv = c_kv
        k = self.w_uk(c_kv).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = self.w_uv(c_kv).view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.wo(y)


class SwiGLUExpert(nn.Module):
    """One expert FFN. SwiGLU = SiLU(x W1) ⊙ (x W3) then W2 — used in DeepSeek/Llama family."""

    def __init__(self, n_embd: int, intermediate: int):
        super().__init__()
        self.w1 = nn.Linear(n_embd, intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, intermediate, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE feed-forward (simplified).

    GPT-2 (llm.c): ONE dense MLP per block.
    DeepSeek-V2: MANY experts; each token only runs top-k experts + shared expert(s).

    Steps per token position:
      1) Router linear → scores over experts
      2) top-k softmax weights
      3) Sum expert outputs weighted
      4) Add shared expert output (always on)
    """

    def __init__(self, config: DeepSeekV2Config):
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.experts = nn.ModuleList(
            [SwiGLUExpert(config.n_embd, config.moe_intermediate) for _ in range(config.n_routed_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLUExpert(config.n_embd, config.moe_intermediate) for _ in range(config.n_shared_experts)]
        )
        self.gate = nn.Linear(config.n_embd, config.n_routed_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → process each (b,t) token
        b, t, c = x.shape
        flat = x.view(-1, c)
        logits = self.gate(flat)
        weights = F.softmax(logits, dim=-1)
        topw, topi = torch.topk(weights, self.topk, dim=-1)
        topw = topw / topw.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat)
        for k in range(self.topk):
            idx = topi[:, k]
            w = topw[:, k].unsqueeze(-1)
            for e_id, expert in enumerate(self.experts):
                mask = idx == e_id
                if mask.any():
                    out[mask] += w[mask] * expert(flat[mask])

        for expert in self.shared:
            out = out + expert(flat)

        return out.view(b, t, c)


class DeepSeekV2Block(nn.Module):
    """One transformer block: MLA + MoE (both pre-norm with RMSNorm)."""

    def __init__(self, config: DeepSeekV2Config):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.attn = MultiHeadLatentAttention(config)
        self.ln2 = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.moe = DeepSeekMoE(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x


class DeepSeekV2(nn.Module):
    """Full tiny language model — compare to llmc.model.GPT in notebook 6."""

    def __init__(self, config: DeepSeekV2Config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([DeepSeekV2Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def kv_cache_bytes_per_token(self) -> int:
        """Educational: MLA latent cache size (float32) per token per layer."""
        return self.config.n_layer * self.config.kv_lora_rank * 4

    def mha_kv_cache_bytes_per_token(self) -> int:
        """What GPT-2-style MHA would store per token per layer (for comparison)."""
        head_dim = self.config.n_embd // self.config.n_head
        return self.config.n_layer * 2 * self.config.n_head * head_dim * 4
