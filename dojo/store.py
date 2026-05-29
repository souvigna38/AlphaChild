"""SQLite persistence for tiers and completed lessons."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "dojo_state.sqlite"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hardware (
            user_id TEXT PRIMARY KEY,
            tier TEXT NOT NULL,
            profile_blob TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS completed (
            user_id TEXT NOT NULL,
            lesson_id TEXT NOT NULL,
            PRIMARY KEY (user_id, lesson_id)
        )
        """
    )
    conn.commit()
    return conn


def set_hardware_tier(user_id: int, tier: str, profile_blob: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO hardware (user_id, tier, profile_blob) VALUES (?, ?, ?)",
            (str(user_id), tier, profile_blob),
        )
        conn.commit()


def get_hardware_tier(user_id: int) -> tuple[str, str] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT tier, profile_blob FROM hardware WHERE user_id = ?",
            (str(user_id),),
        ).fetchone()
    if row is None:
        return None
    return row[0], row[1]


def mark_lesson_complete(user_id: int, lesson_id: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO completed (user_id, lesson_id) VALUES (?, ?)",
            (str(user_id), lesson_id.upper()),
        )
        conn.commit()


def completed_lessons(user_id: int) -> set[str]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT lesson_id FROM completed WHERE user_id = ?",
            (str(user_id),),
        ).fetchall()
    return {r[0] for r in rows}
