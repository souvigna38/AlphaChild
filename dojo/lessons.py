"""Lesson registry: pytest gates, Discord roles, hardware tiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DOJO_ROOT = Path(__file__).resolve().parent
LESSONS_FILE = DOJO_ROOT / "lessons.yaml"


@dataclass(frozen=True)
class Lesson:
    lesson_id: str
    title: str
    track: str
    role: str
    pytest_target: str
    min_tier: str
    notebook: str = ""

    @property
    def gate_path(self) -> Path:
        return DOJO_ROOT / "gates" / self.pytest_target


def load_lessons() -> dict[str, Lesson]:
    raw = yaml.safe_load(LESSONS_FILE.read_text(encoding="utf-8"))
    out: dict[str, Lesson] = {}
    for track_key, lessons in raw.get("tracks", {}).items():
        for entry in lessons:
            lid = entry["id"].upper()
            out[lid] = Lesson(
                lesson_id=lid,
                title=entry["title"],
                track=track_key,
                role=entry.get("role", lid),
                pytest_target=entry["gate"],
                min_tier=entry.get("min_tier", "lightweight"),
                notebook=entry.get("notebook", ""),
            )
    return out


def get_lesson(lesson_id: str) -> Lesson | None:
    return load_lessons().get(lesson_id.upper())


def lessons_for_track(track: str) -> list[Lesson]:
    return [L for L in load_lessons().values() if L.track == track]


def prerequisite_met(completed: set[str], lesson_id: str) -> bool:
    """Enforce linear unlock within track (lesson N requires N-1)."""
    lessons = load_lessons()
    lesson = lessons.get(lesson_id.upper())
    if lesson is None:
        return False
    track_lessons = sorted(
        [L for L in lessons.values() if L.track == lesson.track],
        key=lambda x: x.lesson_id,
    )
    idx = next((i for i, L in enumerate(track_lessons) if L.lesson_id == lesson.lesson_id), -1)
    if idx <= 0:
        return True
    prev = track_lessons[idx - 1].lesson_id
    return prev in completed
