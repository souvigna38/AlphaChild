#!/usr/bin/env python3
"""
Compare C hash_moe token forward vs llmc HashMoE (PyTorch) with shared random weights.
Runs on Mac CPU/MPS — no CUDA required.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llmc.deepseek_v4 import DeepSeekV4Config, HashMoE, build_hash_routing_table

C_DIR = ROOT / "c"
LIB = C_DIR / "bin" / "libhash_moe_check.so"


def _build_c_lib() -> None:
    subprocess.run(
        [
            "cc",
            "-O2",
            "-shared",
            "-fPIC",
            "-o",
            str(LIB),
            str(C_DIR / "hash_moe_check.c"),
            str(C_DIR / "deepseek_v4_config.c"),
            str(C_DIR / "rmsnorm.c"),
            str(C_DIR / "v4_ops.c"),
            str(C_DIR / "rope.c"),
            str(C_DIR / "swiglu.c"),
            str(C_DIR / "sliding_attn.c"),
            str(C_DIR / "ds4_cuda.c"),
            str(C_DIR / "hash_moe.c"),
            "-I",
            str(C_DIR),
            "-lm",
        ],
        check=True,
        cwd=C_DIR,
    )


def main() -> None:
    cfg = DeepSeekV4Config(hidden_size=64, moe_intermediate_size=96, vocab_size=512)
    C = cfg.hidden_size
    E = cfg.n_routed_experts
    I = cfg.moe_intermediate_size
    k = cfg.num_experts_per_tok
    S = cfg.n_shared_experts

    torch.manual_seed(42)
    moe = HashMoE(cfg)
    moe.eval()

    x = torch.randn(C)
    tid = 17
    ids = torch.tensor([[tid]], dtype=torch.long)

    with torch.no_grad():
        y_py = moe(x.unsqueeze(0).unsqueeze(0), ids).squeeze()

    _build_c_lib()
    lib = ctypes.CDLL(str(LIB))
    lib.hash_moe_check_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]

    gate_w = moe.gate.weight.detach().cpu().numpy().astype(np.float32)
    tid2eid = build_hash_routing_table(cfg).cpu().numpy().astype(np.int32)

    expert_gu = np.stack([e.gate_up.weight.detach().cpu().numpy() for e in moe.experts], axis=0).astype(
        np.float32
    )
    expert_dn = np.stack([e.down.weight.detach().cpu().numpy() for e in moe.experts], axis=0).astype(np.float32)
    shared_gu = np.stack([s.gate_up.weight.detach().cpu().numpy() for s in moe.shared], axis=0).astype(np.float32)
    shared_dn = np.stack([s.down.weight.detach().cpu().numpy() for s in moe.shared], axis=0).astype(np.float32)

    x_np = x.detach().cpu().numpy().astype(np.float32)
    out_c = np.zeros(C, dtype=np.float32)

    lib.hash_moe_check_forward(
        out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tid,
        gate_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        expert_gu.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        expert_dn.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        shared_gu.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        shared_dn.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tid2eid.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )

    err = float(np.max(np.abs(out_c - y_py.cpu().numpy())))
    print("hash_moe C vs llmc max_abs_err=%.6g" % err)
    if err > 5e-3:
        print("FAIL", file=sys.stderr)
        sys.exit(1)
    print("OK — hash_moe token forward (C vs PyTorch llmc)")


if __name__ == "__main__":
    main()
