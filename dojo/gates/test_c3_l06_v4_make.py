"""Gate C3-L06 — full C test suite (requires gcc/make)."""

import shutil
import subprocess
from pathlib import Path

import pytest

C_ROOT = Path(__file__).resolve().parents[2] / "llm-c-from-scratch" / "c"


@pytest.mark.slow
def test_make_test_v4():
    if not shutil.which("make") or not shutil.which("gcc"):
        pytest.skip("gcc/make not available")
    proc = subprocess.run(
        ["make", "test_v4"],
        cwd=str(C_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
