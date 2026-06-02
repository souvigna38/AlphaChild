"""
DeepSeek-V4 educational PyTorch (tiny) — mirrors vendor/nano-deepseek-v4.

Use this notebook-side; full model forward is delegated to nano when installed.
See docs/V4_SOURCES_AND_SCOPE.md for the C port phases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DeepSeekV4Config:
    """Small defaults aligned with c/deepseek_v4_config.c and nano_deepseek_v4."""

    vocab_size: int = 512
    hidden_size: int = 64
    moe_intermediate_size: int = 96
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    head_dim: int = 16
    q_lora_rank: int = 32
    num_experts_per_tok: int = 2
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    num_hash_layers: int = 3
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 8
    sliding_window: int = 8
    compress_rate_csa: int = 4
    compress_rate_hca: int = 128
    index_n_heads: int = 4
    index_head_dim: int = 8
    index_topk: int = 4
    o_groups: int = 2
    o_lora_rank: int = 16
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    layer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None

    def __post_init__(self) -> None:
        if self.layer_types is None:
            interleave = [
                "compressed_sparse_attention" if i % 2 == 0 else "heavily_compressed_attention"
                for i in range(max(self.num_hidden_layers - 2, 0))
            ]
            self.layer_types = ["sliding_attention"] * min(self.num_hidden_layers, 2) + interleave
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "hash_moe" if i < self.num_hash_layers else "moe" for i in range(self.num_hidden_layers)
            ]

    @classmethod
    def tiny(cls, vocab_size: int = 64, block_size: int = 32) -> DeepSeekV4Config:
        """Fast config for notebooks, pytest, and Dojo gate C2-L18 (Mac-friendly)."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=32,
            moe_intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            q_lora_rank=16,
            num_experts_per_tok=2,
            n_routed_experts=4,
            n_shared_experts=1,
            num_hash_layers=2,
            hc_mult=2,
            sliding_window=8,
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class SwiGLUExpert(nn.Module):
    """Same layout as nano: fused gate_up + down, with clamps."""

    def __init__(self, cfg: DeepSeekV4Config) -> None:
        super().__init__()
        self.limit = cfg.swiglu_limit
        self.gate_up = nn.Linear(cfg.hidden_size, 2 * cfg.moe_intermediate_size, bias=False)
        self.down = nn.Linear(cfg.moe_intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return self.down(F.silu(gate) * up)


def build_hash_routing_table(cfg: DeepSeekV4Config) -> torch.Tensor:
    """tid2eid (vocab, topk) — matches c/hash_moe.c and nano."""
    token_ids = torch.arange(cfg.vocab_size).unsqueeze(1)
    offsets = torch.arange(cfg.num_experts_per_tok).unsqueeze(0)
    return (token_ids * 1103515245 + 12345 + offsets * 2654435761).remainder(cfg.n_routed_experts)


class HashMoE(nn.Module):
    """Bootstrap hash-MoE: token-id routing + sqrt(softplus(gate)) weights."""

    def __init__(self, cfg: DeepSeekV4Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.hidden_size, cfg.n_routed_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(cfg) for _ in range(cfg.n_routed_experts)])
        self.shared = nn.ModuleList([SwiGLUExpert(cfg) for _ in range(cfg.n_shared_experts)])
        self.register_buffer("tid2eid", build_hash_routing_table(cfg), persistent=False)

    def route(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(hidden_states)
        affinity = torch.sqrt(F.softplus(logits))
        topk_idx = self.tid2eid[input_ids]
        weights = affinity.gather(-1, topk_idx)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights * self.cfg.routed_scaling_factor
        return topk_idx, weights

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        b, t, h = hidden_states.shape
        topk_idx, weights = self.route(hidden_states, input_ids)
        flat_h = hidden_states.reshape(-1, h)
        flat_idx = topk_idx.reshape(-1, topk_idx.shape[-1])
        flat_w = weights.reshape(-1, weights.shape[-1])
        out = flat_h.new_zeros(flat_h.shape)
        for e, expert in enumerate(self.experts):
            mask = flat_idx == e
            tok_mask = mask.any(dim=-1)
            if not tok_mask.any():
                continue
            w = (flat_w[tok_mask] * mask[tok_mask].to(flat_w.dtype)).sum(dim=-1, keepdim=True)
            out[tok_mask] += expert(flat_h[tok_mask]) * w
        for shared in self.shared:
            out = out + shared(flat_h)
        return out.view(b, t, h)


class RoutedMoE(nn.Module):
    """Routed experts: sqrt(softplus(gate)) + top-k + e_score_correction_bias."""

    def __init__(self, cfg: DeepSeekV4Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.hidden_size, cfg.n_routed_experts, bias=False)
        self.route_bias = nn.Parameter(torch.zeros(cfg.n_routed_experts))
        self.experts = nn.ModuleList([SwiGLUExpert(cfg) for _ in range(cfg.n_routed_experts)])
        self.shared = nn.ModuleList([SwiGLUExpert(cfg) for _ in range(cfg.n_shared_experts)])

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        del input_ids
        b, t, h = hidden_states.shape
        flat_h = hidden_states.reshape(-1, h)
        logits = self.gate(flat_h)
        affinity = torch.sqrt(F.softplus(logits))
        scores = affinity + self.route_bias
        topk_w, topk_idx = torch.topk(scores, self.cfg.num_experts_per_tok, dim=-1)
        topk_w = affinity.gather(-1, topk_idx)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        topk_w = topk_w * self.cfg.routed_scaling_factor
        out = flat_h.new_zeros(flat_h.shape)
        for e, expert in enumerate(self.experts):
            mask = topk_idx == e
            tok_mask = mask.any(dim=-1)
            if not tok_mask.any():
                continue
            w = (topk_w[tok_mask] * mask[tok_mask].to(topk_w.dtype)).sum(dim=-1, keepdim=True)
            out[tok_mask] += expert(flat_h[tok_mask]) * w
        for shared in self.shared:
            out = out + shared(flat_h)
        return out.view(b, t, h)


def load_nano_model(cfg: DeepSeekV4Config | None = None):
    """Optional: return nano_deepseek_v4.DeepSeekV4Model for parity checks."""
    from nano_deepseek_v4.config import DeepSeekV4Config as NanoCfg
    from nano_deepseek_v4.modeling import DeepSeekV4Model

    if cfg is None:
        return DeepSeekV4Model(NanoCfg())
    nano = NanoCfg(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        moe_intermediate_size=cfg.moe_intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_hash_layers=cfg.num_hash_layers,
        n_routed_experts=cfg.n_routed_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        n_shared_experts=cfg.n_shared_experts,
    )
    return DeepSeekV4Model(nano)
