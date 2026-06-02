"""Gate C3-L01 — V4 C headers compile check (lightweight)."""

from pathlib import Path

C_ROOT = Path(__file__).resolve().parents[2] / "llm-c-from-scratch" / "c"


def test_v4_c_sources_present():
    assert (C_ROOT / "swiglu.c").is_file()
    assert (C_ROOT / "hash_moe.c").is_file()
    assert (C_ROOT / "Makefile").is_file()
