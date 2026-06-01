"""Read/write dojo/.env without extra dependencies."""

from __future__ import annotations

import re
from pathlib import Path

DOJO_DIR = Path(__file__).resolve().parent
ENV_PATH = DOJO_DIR / ".env"
ENV_EXAMPLE = DOJO_DIR / ".env.example"


def parse_env(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            out[key.strip()] = val.strip()
    return out


def load_env(path: Path | None = None) -> dict[str, str]:
    p = path or ENV_PATH
    if not p.is_file():
        return {}
    return parse_env(p.read_text(encoding="utf-8"))


def merge_env(updates: dict[str, str], path: Path | None = None) -> Path:
    """Merge keys into .env, creating from .env.example if missing."""
    p = path or ENV_PATH
    if p.is_file():
        lines = p.read_text(encoding="utf-8").splitlines()
        existing_keys = set()
        new_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                existing_keys.add(key)
                if key in updates:
                    new_lines.append(f"{key}={updates[key]}")
                    del updates[key]
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        for key, val in updates.items():
            if key not in existing_keys:
                new_lines.append(f"{key}={val}")
        p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    elif ENV_EXAMPLE.is_file():
        text = ENV_EXAMPLE.read_text(encoding="utf-8")
        for key, val in updates.items():
            text = re.sub(
                rf"^{re.escape(key)}=.*$",
                f"{key}={val}",
                text,
                flags=re.MULTILINE,
            )
        p.write_text(text, encoding="utf-8")
    else:
        p.write_text("\n".join(f"{k}={v}" for k, v in updates.items()) + "\n", encoding="utf-8")
    return p
