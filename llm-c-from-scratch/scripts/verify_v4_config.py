#!/usr/bin/env python3
"""Check C tiny config matches nano-deepseek-v4 defaults."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "nano-deepseek-v4"
C_DIR = ROOT / "c"
BIN = C_DIR / "bin" / "train_deepseek_v4_tiny"


def c_config_fields() -> dict[str, int | float]:
    if not BIN.exists():
        subprocess.run(["make", "-C", str(C_DIR), "bin/train_deepseek_v4_tiny"], check=True)
    out = subprocess.run([str(BIN)], cwd=C_DIR, capture_output=True, text=True, check=True)
    fields: dict[str, int | float] = {}
    for line in out.stdout.splitlines():
        if "hidden_size=" in line:
            parts = line.split()
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    fields[k.strip()] = int(v)
        if "compress:" in line:
            for token in line.replace("compress:", "").split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    fields[k] = int(v)
    return fields


def nano_config_fields() -> dict[str, int | float]:
    if not VENDOR.exists():
        raise FileNotFoundError("vendor/nano-deepseek-v4 missing; run ./scripts/setup_vendor.sh")
    sys.path.insert(0, str(VENDOR))
    from nano_deepseek_v4.config import DeepSeekV4Config

    cfg = DeepSeekV4Config()
    return {
        "hidden_size": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
        "heads": cfg.num_attention_heads,
        "head_dim": cfg.head_dim,
        "csa_rate": cfg.compress_rates["compressed_sparse_attention"],
        "hca_rate": cfg.compress_rates["heavily_compressed_attention"],
        "index_topk": cfg.index_topk,
    }


def main() -> None:
    c = c_config_fields()
    n = nano_config_fields()
    checks = [
        ("hidden_size", c.get("hidden_size"), n["hidden_size"]),
        ("layers", c.get("layers"), n["layers"]),
        ("heads", c.get("heads"), n["heads"]),
        ("head_dim", c.get("head_dim"), n["head_dim"]),
        ("csa_rate", c.get("csa_rate"), n["csa_rate"]),
        ("hca_rate", c.get("hca_rate"), n["hca_rate"]),
        ("index_topk", c.get("index_topk"), n["index_topk"]),
    ]
    bad = [(name, cv, nv) for name, cv, nv in checks if cv != nv]
    if bad:
        for name, cv, nv in bad:
            print(f"MISMATCH {name}: C={cv} nano={nv}", file=sys.stderr)
        sys.exit(1)
    print("C tiny config matches nano-deepseek-v4 defaults")


if __name__ == "__main__":
    main()
