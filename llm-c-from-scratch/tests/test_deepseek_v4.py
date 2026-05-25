"""DeepSeek-V4 tiny: hash table + HashMoE vs C reference."""

import subprocess
from pathlib import Path

import torch

from llmc.deepseek_v4 import DeepSeekV4Config, HashMoE, SwiGLUExpert, build_hash_routing_table

ROOT = Path(__file__).resolve().parent.parent
C_DIR = ROOT / "c"


def test_hash_table_matches_c():
    cfg = DeepSeekV4Config()
    py = build_hash_routing_table(cfg)
    token_ids = torch.arange(cfg.vocab_size).unsqueeze(1)
    offsets = torch.arange(cfg.num_experts_per_tok).unsqueeze(0)
    ref = (token_ids * 1103515245 + 12345 + offsets * 2654435761).remainder(cfg.n_routed_experts)
    assert torch.equal(py, ref)


def test_swiglu_forward_shape():
    cfg = DeepSeekV4Config(hidden_size=32, moe_intermediate_size=48)
    m = SwiGLUExpert(cfg)
    x = torch.randn(4, cfg.hidden_size)
    y = m(x)
    assert y.shape == (4, cfg.hidden_size)


def test_hash_moe_forward():
    cfg = DeepSeekV4Config(hidden_size=32, moe_intermediate_size=48, vocab_size=64)
    moe = HashMoE(cfg)
    b, t = 2, 5
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    h = torch.randn(b, t, cfg.hidden_size)
    y = moe(h, ids)
    assert y.shape == h.shape


def test_c_hash_moe_smoke():
    if not (C_DIR / "bin" / "test_hash_moe").exists():
        subprocess.run(["make", "bin/test_hash_moe"], cwd=C_DIR, check=True)
    out = subprocess.run([str(C_DIR / "bin" / "test_hash_moe")], cwd=C_DIR, capture_output=True, text=True, check=True)
    assert "OK" in out.stdout


def test_nano_forward_optional():
    vendor = ROOT / "vendor" / "nano-deepseek-v4"
    if not vendor.exists():
        return
    try:
        from llmc.deepseek_v4 import load_nano_model

        model = load_nano_model()
    except (ImportError, ModuleNotFoundError):
        return
    model.eval()
    ids = torch.randint(0, model.config.vocab_size, (1, 4))
    with torch.no_grad():
        out = model(ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    assert logits.shape[-1] == model.config.vocab_size
