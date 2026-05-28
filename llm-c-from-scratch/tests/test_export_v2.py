"""Export + C checkpoint load smoke test."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "v2_tiny_test.bin"
TRAIN_BIN = ROOT / "c" / "bin" / "train_v2_tiny"


@pytest.mark.skipif(not (ROOT / "c" / "Makefile").exists(), reason="no C tree")
def test_export_and_c_sample():
    subprocess.run(
        [
            "python3",
            str(ROOT / "scripts" / "export_v2_tiny.py"),
            "--match-train-c",
            "--train-steps",
            "5",
            "-o",
            str(CKPT),
        ],
        cwd=ROOT,
        check=True,
    )
    if not TRAIN_BIN.exists():
        subprocess.run(["make", "bin/train_v2_tiny"], cwd=ROOT / "c", check=True)
    r = subprocess.run(
        [str(TRAIN_BIN), "-sample", "-ckpt", str(CKPT)],
        cwd=ROOT / "c",
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "sample" in r.stdout.lower() or "ROMEO" in r.stdout
