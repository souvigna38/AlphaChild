#!/usr/bin/env python3
"""
Compare C ds4_sliding_attn_forward vs nano DeepSeekV4Attention (sliding, no cache).
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
C_DIR = ROOT / "c"
VENDOR = ROOT / "vendor" / "nano-deepseek-v4"
LIB = C_DIR / "bin" / "libsliding_attn_check.so"


def _build_c_lib() -> None:
    subprocess.run(
        [
            "cc",
            "-O2",
            "-shared",
            "-fPIC",
            "-o",
            str(LIB),
            str(C_DIR / "sliding_attn_check.c"),
            str(C_DIR / "deepseek_v4_config.c"),
            str(C_DIR / "rmsnorm.c"),
            str(C_DIR / "v4_ops.c"),
            str(C_DIR / "rope.c"),
            str(C_DIR / "swiglu.c"),
            str(C_DIR / "sliding_attn.c"),
            str(C_DIR / "ds4_cuda.c"),
            "-I",
            str(C_DIR),
            "-lm",
        ],
        check=True,
        cwd=C_DIR,
    )


def _tiny_cfg():
    from nano_deepseek_v4.config import DeepSeekV4Config

    return DeepSeekV4Config(
        vocab_size=512,
        hidden_size=64,
        moe_intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        head_dim=16,
        q_lora_rank=32,
        sliding_window=8,
        o_groups=2,
        o_lora_rank=16,
        partial_rotary_factor=0.5,
        attention_dropout=0.0,
    )


def main() -> None:
    if not VENDOR.exists():
        print("Run: ./scripts/setup_vendor.sh", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(VENDOR))
    from nano_deepseek_v4.modeling import DeepSeekV4Attention

    cfg = _tiny_cfg()
    attn = DeepSeekV4Attention(cfg, layer_type="sliding_attention")
    attn.eval()

    T = 8
    C = cfg.hidden_size
    r = cfg.q_lora_rank
    D = cfg.head_dim
    NH = cfg.num_attention_heads
    attn_w = NH * D
    o_mid = cfg.o_groups * cfg.o_lora_rank
    in_pg = attn_w // cfg.o_groups

    torch.manual_seed(123)
    x_t = torch.randn(1, T, C)
    pos = torch.arange(T, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        y_py = attn(x_t, pos, attention_mask=None, cache=None).squeeze(0)

    wq_a = attn.q_a_proj.weight.detach().cpu().numpy().astype(np.float32)
    wqa_n = attn.q_a_norm.weight.detach().cpu().numpy().astype(np.float32)
    wq_b = attn.q_b_proj.weight.detach().cpu().numpy().astype(np.float32)
    wkv = attn.kv_proj.weight.detach().cpu().numpy().astype(np.float32)
    wkvn = attn.kv_norm.weight.detach().cpu().numpy().astype(np.float32)
    sink = attn.attention_sink.detach().cpu().numpy().astype(np.float32)
    wo_a = attn.o_a_proj.weight.detach().cpu().numpy().astype(np.float32).reshape(-1)
    wo_b = attn.o_b_proj.weight.detach().cpu().numpy().astype(np.float32)

    scratch_n = (
        T * r
        + NH * T * D
        + T * D
        + T * (cfg.qk_rope_head_dim // 2) * 2
        + NH * T * D * 2
        + T * attn_w
        + T * o_mid
        + attn_w
    )
    scratch = np.zeros(scratch_n, dtype=np.float32)
    x_np = x_t.squeeze(0).numpy().astype(np.float32)
    out_c = np.zeros((T, C), dtype=np.float32)

    _build_c_lib()
    lib = ctypes.CDLL(str(LIB))
    lib.sliding_attn_check_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]

    lib.sliding_attn_check_forward(
        out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        T,
        wq_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wqa_n.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wq_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wkv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wkvn.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sink.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wo_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wo_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    err = float(np.max(np.abs(out_c - y_py.cpu().numpy())))
    print("sliding_attn C vs nano max_abs_err=%.6g" % err)
    if err > 1e-3:
        t_worst = int(np.argmax(np.abs(out_c - y_py.cpu().numpy()).sum(axis=1)))
        print("  worst token", t_worst, "py", y_py[t_worst, :4].tolist(), "c", out_c[t_worst, :4].tolist())
        sys.exit(1)
    print("OK — sliding_attn forward matches nano (T=%d C=%d)" % (T, C))


if __name__ == "__main__":
    main()
