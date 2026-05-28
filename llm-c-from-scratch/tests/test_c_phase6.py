"""Phase 6: AdamW train, save/load checkpoint round-trip."""

import re
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
C_DIR = ROOT / "c"
BIN = C_DIR / "bin" / "train_v2_tiny"
DSV2_MAGIC = 0x32565344


def test_adam_train_and_save_load_roundtrip():
    if not BIN.exists():
        subprocess.run(["make", "bin/train_v2_tiny"], cwd=C_DIR, check=True)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "v2_c_phase6.bin"
        out = subprocess.run(
            [str(BIN), "-train-adam", "12", "-batch", "2", "-lr", "0.003", "-save", str(ckpt)],
            cwd=C_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        losses = [float(x) for x in re.findall(r"train step \d+ loss ([0-9.+-eE]+)", out.stdout)]
        assert len(losses) >= 2
        assert all(l == l and abs(l) < 50.0 for l in losses)
        assert ckpt.exists() and ckpt.stat().st_size > 100

        magic = struct.unpack("I", ckpt.read_bytes()[:4])[0]
        assert magic == DSV2_MAGIC

        sample = subprocess.run(
            [str(BIN), "-sample", "-ckpt", str(ckpt)],
            cwd=C_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "sample" in sample.stdout.lower() or "ROMEO" in sample.stdout
