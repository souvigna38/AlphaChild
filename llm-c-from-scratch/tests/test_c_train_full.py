"""Smoke test: Phase 5b C trainer (-train-full) stays finite."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C_DIR = ROOT / "c"
BIN = C_DIR / "bin" / "train_v2_tiny"


def test_train_full_smoke():
    if not BIN.exists():
        subprocess.run(["make", "bin/train_v2_tiny"], cwd=C_DIR, check=True)
    out = subprocess.run(
        [str(BIN), "-train-full", "8"],
        cwd=C_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    losses = [float(x) for x in re.findall(r"full step \d+ loss ([0-9.+-eE]+)", out.stdout)]
    assert len(losses) >= 2
    assert all(l == l and abs(l) < 100.0 for l in losses), losses
