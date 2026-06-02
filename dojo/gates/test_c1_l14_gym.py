"""Gate C1-L14 — Gymnasium available (Atari track optional)."""

import pytest

pytest.importorskip("gymnasium")


def test_gymnasium_import():
    import gymnasium as gym

    assert hasattr(gym, "make")
