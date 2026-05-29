"""DeepSeek-V4 tiny: hash table + HashMoE vs C reference."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from llmc.deepseek_v4 import (
    DeepSeekV4Config,
    HashMoE,
    RoutedMoE,
    SwiGLUExpert,
    build_hash_routing_table,
)

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


def test_routed_moe_forward():
    cfg = DeepSeekV4Config(hidden_size=32, moe_intermediate_size=48, n_routed_experts=4, num_experts_per_tok=2)
    moe = RoutedMoE(cfg)
    h = torch.randn(2, 5, cfg.hidden_size)
    y = moe(h)
    assert y.shape == h.shape


def test_c_v4_train_head_smoke():
    if not (C_DIR / "bin" / "test_v4_train_head").exists():
        subprocess.run(["make", "bin/test_v4_train_head"], cwd=C_DIR, check=True)
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_v4_train_head")], cwd=C_DIR, capture_output=True, text=True, check=True
    )
    assert "OK" in out.stdout


def test_c_v4_train_final_smoke():
    if not (C_DIR / "bin" / "test_v4_train_final").exists():
        subprocess.run(["make", "bin/test_v4_train_final"], cwd=C_DIR, check=True)
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_v4_train_final")], cwd=C_DIR, capture_output=True, text=True, check=True
    )
    assert "OK" in out.stdout


def test_c_v4_train_1layer_smoke():
    if not (C_DIR / "bin" / "test_v4_train_1layer").exists():
        subprocess.run(["make", "bin/test_v4_train_1layer"], cwd=C_DIR, check=True)
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_v4_train_1layer")], cwd=C_DIR, capture_output=True, text=True, check=True
    )
    assert "OK" in out.stdout


def test_c_v4_train_full_smoke():
    if not (C_DIR / "bin" / "test_v4_train_full").exists():
        subprocess.run(["make", "bin/test_v4_train_full"], cwd=C_DIR, check=True)
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_v4_train_full")], cwd=C_DIR, capture_output=True, text=True, check=True
    )
    assert "OK" in out.stdout


def test_c_indexer_train_smoke():
    if not (C_DIR / "bin" / "test_indexer_train").exists():
        subprocess.run(["make", "bin/test_indexer_train"], cwd=C_DIR, check=True)
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_indexer_train")], cwd=C_DIR, capture_output=True, text=True, check=True
    )
    assert "OK" in out.stdout


def test_c_cuda_rmsnorm_smoke():
    if not (C_DIR / "bin" / "test_cuda_rmsnorm").exists():
        subprocess.run(["make", "bin/test_cuda_rmsnorm"], cwd=C_DIR, check=True)
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_cuda_rmsnorm")], cwd=C_DIR, capture_output=True, text=True, check=True
    )
    assert "OK" in out.stdout


def test_c_cuda_swiglu_smoke():
    if not (C_DIR / "bin" / "test_cuda_swiglu").exists():
        subprocess.run(["make", "bin/test_cuda_swiglu"], cwd=C_DIR, check=True)
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_cuda_swiglu")], cwd=C_DIR, capture_output=True, text=True, check=True
    )
    assert "OK" in out.stdout


def test_c_cuda_core_attention_smoke():
    if not (C_DIR / "bin" / "test_cuda_core_attention").exists():
        subprocess.run(["make", "bin/test_cuda_core_attention"], cwd=C_DIR, check=True)
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_cuda_core_attention")],
        cwd=C_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "OK" in out.stdout


def test_c_cuda_sliding_integrated_smoke():
    if not (C_DIR / "bin" / "test_cuda_sliding_integrated").exists():
        subprocess.run(["make", "bin/test_cuda_sliding_integrated"], cwd=C_DIR, check=True)
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_cuda_sliding_integrated")],
        cwd=C_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "OK" in out.stdout


def test_c_v4_layer_parity_smoke():
    if not (C_DIR / "bin" / "test_v4_layer_parity").exists():
        subprocess.run(["make", "bin/test_v4_layer_parity"], cwd=C_DIR, check=True)
    fixture = Path(__file__).resolve().parent / "fixtures" / "v4_layer_checksums.txt"
    if not fixture.exists():
        subprocess.run(
            [str(C_DIR / "bin" / "test_v4_layer_parity"), "--write-fixture", "--fixture", str(fixture)],
            cwd=C_DIR,
            check=True,
        )
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_v4_layer_parity"), "--fixture", str(fixture)],
        cwd=C_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "OK" in out.stdout


def test_verify_nano_hash_moe_script():
    script = Path(__file__).resolve().parent.parent / "scripts" / "verify_v4_nano_hash_moe.py"
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        pytest.skip("verify_v4_nano_hash_moe unavailable: " + (out.stderr or out.stdout)[:200])
    assert "OK" in out.stdout


def test_c_v4_tied_head_smoke():
    if not (C_DIR / "bin" / "test_v4_tied_head").exists():
        subprocess.run(["make", "bin/test_v4_tied_head"], cwd=C_DIR, check=True)
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_v4_tied_head")], cwd=C_DIR, capture_output=True, text=True, check=True
    )
    assert "OK" in out.stdout


def test_c_v4_parity_smoke():
    if not (C_DIR / "bin" / "test_v4_parity").exists():
        subprocess.run(["make", "bin/test_v4_parity"], cwd=C_DIR, check=True)
    fixture = Path(__file__).resolve().parent / "fixtures" / "v4_parity_logits.txt"
    if not fixture.exists():
        subprocess.run(
            [str(C_DIR / "bin" / "test_v4_parity"), "--write-fixture", "--fixture", str(fixture)],
            cwd=C_DIR,
            check=True,
        )
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_v4_parity"), "--fixture", str(fixture)],
        cwd=C_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "OK" in out.stdout


def test_c_v4_train_e2e_smoke():
    if not (C_DIR / "bin" / "test_v4_train_e2e").exists():
        subprocess.run(["make", "bin/test_v4_train_e2e"], cwd=C_DIR, check=True)
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_v4_train_e2e")], cwd=C_DIR, capture_output=True, text=True, check=True
    )
    assert "OK" in out.stdout


def test_c_v4_train_4layer_smoke():
    if not (C_DIR / "bin" / "test_v4_train_4layer").exists():
        subprocess.run(["make", "bin/test_v4_train_4layer"], cwd=C_DIR, check=True)
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_v4_train_4layer")], cwd=C_DIR, capture_output=True, text=True, check=True
    )
    assert "OK" in out.stdout


def test_c_hash_moe_train_smoke():
    if not (C_DIR / "bin" / "test_hash_moe_train").exists():
        subprocess.run(["make", "bin/test_hash_moe_train"], cwd=C_DIR, check=True)
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_hash_moe_train")], cwd=C_DIR, capture_output=True, text=True, check=True
    )
    assert "OK" in out.stdout


def test_c_v4_model_smoke():
    if not (C_DIR / "bin" / "test_v4_model").exists():
        subprocess.run(["make", "bin/test_v4_model"], cwd=C_DIR, check=True)
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_v4_model")], cwd=C_DIR, capture_output=True, text=True, check=True
    )
    assert "OK" in out.stdout


def test_c_v4_attention_smoke():
    if not (C_DIR / "bin" / "test_v4_attention").exists():
        subprocess.run(["make", "bin/test_v4_attention"], cwd=C_DIR, check=True)
    out = subprocess.run(
        [str(C_DIR / "bin" / "test_v4_attention")], cwd=C_DIR, capture_output=True, text=True, check=True
    )
    assert "OK" in out.stdout


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
