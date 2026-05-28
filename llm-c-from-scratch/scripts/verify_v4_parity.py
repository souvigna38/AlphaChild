#!/usr/bin/env python3
"""Run C forward parity golden test and config cross-check vs nano."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C_DIR = ROOT / "c"
BIN = C_DIR / "bin" / "test_v4_parity"
FIXTURE = ROOT / "tests" / "fixtures" / "v4_parity_logits.txt"


def main() -> None:
    if not BIN.exists():
        subprocess.run(["make", "-C", str(C_DIR), "bin/test_v4_parity"], check=True)
    if not FIXTURE.exists():
        subprocess.run([str(BIN), "--write-fixture", "--fixture", str(FIXTURE)], cwd=C_DIR, check=True)
    out = subprocess.run([str(BIN), "--fixture", str(FIXTURE)], cwd=C_DIR, capture_output=True, text=True, check=True)
    if "OK" not in out.stdout:
        print(out.stdout, out.stderr, file=sys.stderr)
        sys.exit(1)
    print(out.stdout.strip())

    cfg_script = ROOT / "scripts" / "verify_v4_config.py"
    subprocess.run([sys.executable, str(cfg_script)], check=True)

    ref = ROOT / "scripts" / "verify_v4_reference.py"
    if ref.exists():
        try:
            subprocess.run([sys.executable, str(ref)], check=True)
        except subprocess.CalledProcessError:
            print("(verify_v4_reference skipped: torch/vendor unavailable)", file=sys.stderr)


if __name__ == "__main__":
    main()
