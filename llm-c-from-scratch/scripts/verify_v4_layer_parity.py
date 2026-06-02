#!/usr/bin/env python3
"""Per-layer stream checksum golden test (C deterministic forward)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C_DIR = ROOT / "c"
BIN = C_DIR / "bin" / "test_v4_layer_parity"
FIXTURE = ROOT / "tests" / "fixtures" / "v4_layer_checksums.txt"


def main() -> None:
    if not BIN.exists():
        subprocess.run(["make", "-C", str(C_DIR), "bin/test_v4_layer_parity"], check=True)
    if not FIXTURE.exists():
        subprocess.run(
            [str(BIN), "--write-fixture", "--fixture", str(FIXTURE)],
            cwd=C_DIR,
            check=True,
        )
    out = subprocess.run(
        [str(BIN), "--fixture", str(FIXTURE)],
        cwd=C_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    if "OK" not in out.stdout:
        print(out.stdout, out.stderr, file=sys.stderr)
        sys.exit(1)
    print(out.stdout.strip())


if __name__ == "__main__":
    main()
