#!/usr/bin/env python3
"""Run nano-deepseek-v4 tiny forward (Python ground truth for future C parity tests)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "nano-deepseek-v4"

if not VENDOR.exists():
    print("Run: ./scripts/setup_vendor.sh", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(VENDOR))

try:
    import torch
    from nano_deepseek_v4.config import DeepSeekV4Config
    from nano_deepseek_v4.modeling import DeepSeekV4Model
except ImportError as exc:
    print("Missing deps for nano-deepseek-v4:", exc, file=sys.stderr)
    print("Try: pip install torch safetensors", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    cfg = DeepSeekV4Config()  # small default config for tests
    model = DeepSeekV4Model(cfg)
    model.eval()
    b, t = 2, 8
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    with torch.no_grad():
        out = model(ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    print("nano-deepseek-v4 tiny forward OK")
    print("  hidden=%d layers=%d" % (cfg.hidden_size, cfg.num_hidden_layers))
    print("  logits shape:", tuple(logits.shape))
    print("  layer_types:", cfg.layer_types)
    print("  mlp_layer_types:", cfg.mlp_layer_types)
    print("  For C golden parity: python3 scripts/verify_v4_parity.py")
    print("  For hash_moe C vs llmc: python3 scripts/verify_v4_nano_hash_moe.py")
    with torch.no_grad():
        logits_flat = logits[0, 0].float()
        print("  logits[0,0,:4] =", [float(logits_flat[i]) for i in range(min(4, logits_flat.numel()))])


if __name__ == "__main__":
    main()
