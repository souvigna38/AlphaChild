#!/usr/bin/env python3
"""
Export DeepSeek-V2 tiny weights for C (train_v2_tiny.c -sample -ckpt).

Layout must match c/deepseek_v2/model.c parameter order.
Config must match the defaults in train_v2_tiny.c (or pass --match-train).
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch

from llmc.data import CharTokenizer, load_text
from llmc.deepseek_v2 import DeepSeekV2, DeepSeekV2Config

DSV2_MAGIC = 0x32565344  # 'DSV2'


def config_to_bytes(cfg: DeepSeekV2Config) -> bytes:
    """Binary layout mirrors Dsv2ModelConfig in model.h (11 int32/float fields)."""
    return struct.pack(
        "10if",
        cfg.vocab_size,
        cfg.block_size,
        cfg.n_layer,
        cfg.n_embd,
        cfg.n_head,
        cfg.kv_lora_rank,
        cfg.n_routed_experts,
        cfg.n_shared_experts,
        cfg.num_experts_per_tok,
        cfg.moe_intermediate,
        cfg.rms_norm_eps,
    )


def flatten_model(model: DeepSeekV2) -> list[torch.Tensor]:
    """Same order as dsv2_model_init pointer walk in model.c."""
    cfg = model.config
    L = cfg.n_layer
    tensors: list[torch.Tensor] = [model.tok_emb.weight.detach().cpu()]
    for li in range(L):
        b = model.blocks[li]
        tensors.append(b.ln1.weight.detach().cpu())
        tensors.append(b.ln2.weight.detach().cpu())
        tensors.append(b.attn.wq.weight.detach().cpu())
        tensors.append(b.attn.w_dkv.weight.detach().cpu())
        tensors.append(b.attn.w_uk.weight.detach().cpu())
        tensors.append(b.attn.w_uv.weight.detach().cpu())
        tensors.append(b.attn.wo.weight.detach().cpu())
        tensors.append(b.moe.gate.weight.detach().cpu())
        tensors.append(torch.stack([e.w1.weight for e in b.moe.experts]).detach().cpu())
        tensors.append(torch.stack([e.w2.weight for e in b.moe.experts]).detach().cpu())
        tensors.append(torch.stack([e.w3.weight for e in b.moe.experts]).detach().cpu())
        if b.moe.shared:
            tensors.append(torch.stack([e.w1.weight for e in b.moe.shared]).detach().cpu())
            tensors.append(torch.stack([e.w2.weight for e in b.moe.shared]).detach().cpu())
            tensors.append(torch.stack([e.w3.weight for e in b.moe.shared]).detach().cpu())
    tensors.append(model.ln_f.weight.detach().cpu())
    return tensors


def export_checkpoint(model: DeepSeekV2, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = torch.cat([t.flatten() for t in flatten_model(model)]).numpy().astype("float32")
    with path.open("wb") as f:
        f.write(struct.pack("I", DSV2_MAGIC))
        f.write(config_to_bytes(model.config))
        f.write(flat.tobytes())
    print(f"Wrote {path} ({len(flat)} floats)")


def train_tiny_for_export(cfg: DeepSeekV2Config, train_ids: torch.Tensor, steps: int = 30) -> DeepSeekV2:
    from llmc.train import Trainer, TrainConfig

    model = DeepSeekV2(cfg)
    val = train_ids[: max(1, len(train_ids) // 10)]
    trainer = Trainer(
        model,
        train_ids,
        val,
        TrainConfig(max_steps=steps, batch_size=8, eval_interval=steps, eval_iters=2, learning_rate=3e-3),
    )
    trainer.train()
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", type=Path, default=Path("checkpoints/v2_tiny.bin"))
    p.add_argument("--data", type=Path, default=Path("data/tiny_shakespeare.txt"))
    p.add_argument("--train-steps", type=int, default=40, help="PyTorch steps before export (0 = random)")
    p.add_argument(
        "--match-train-c",
        action="store_true",
        help="Use same hyperparams as train_v2_tiny.c (n_layer=2, n_embd=64, ...)",
    )
    args = p.parse_args()

    text = load_text(args.data)
    tok = CharTokenizer.from_text(text)
    if args.match_train_c:
        cfg = DeepSeekV2Config(
            vocab_size=tok.vocab_size,
            block_size=32,
            n_layer=2,
            n_embd=64,
            n_head=4,
            kv_lora_rank=16,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate=96,
        )
    else:
        cfg = DeepSeekV2Config.tiny(tok.vocab_size, block_size=32)

    train_ids = torch.tensor(tok.encode(text[:8000]), dtype=torch.long)
    if args.train_steps > 0:
        print(f"Training {args.train_steps} steps in PyTorch before export...")
        model = train_tiny_for_export(cfg, train_ids, args.train_steps)
    else:
        model = DeepSeekV2(cfg)
    export_checkpoint(model, args.output)


if __name__ == "__main__":
    main()
