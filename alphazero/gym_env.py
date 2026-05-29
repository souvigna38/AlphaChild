"""Gymnasium environments with the same interface as board-game classes in the notebooks."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None  # type: ignore[assignment]


def _require_gym() -> Any:
    if gym is None:
        raise ImportError(
            "gymnasium is required for Gym environments. "
            "Install with: pip install 'gymnasium[atari,accept-rom-license]'"
        )
    return gym


class GymGame:
    """Wrap a Gymnasium environment so it matches the AlphaZero / MuZero game API."""

    def __init__(
        self,
        env_id: str = "CartPole-v1",
        *,
        frame_stack: int = 1,
        frame_skip: int = 1,
        max_episode_steps: int | None = None,
        render_mode: str | None = None,
    ):
        gym_mod = _require_gym()
        self.env_id = env_id
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self._last_reward = 0.0
        self._last_done = False

        env_kwargs: dict[str, Any] = {}
        if render_mode is not None:
            env_kwargs["render_mode"] = render_mode

        self.env = gym_mod.make(env_id, **env_kwargs)
        if max_episode_steps is not None:
            self.env = gym_mod.wrappers.TimeLimit(self.env, max_episode_steps=max_episode_steps)

        self.action_size = int(self.env.action_space.n)
        obs_space = self.env.observation_space

        if len(obs_space.shape) == 1:
            self.obs_shape = (obs_space.shape[0],)
            self.obs_kind = "vector"
        elif len(obs_space.shape) == 3:
            self.obs_shape = obs_space.shape
            self.obs_kind = "image"
        else:
            raise ValueError(f"Unsupported observation space: {obs_space}")

        if self.obs_kind == "image":
            self.row_count = 6
            self.column_count = 6
        else:
            self.row_count = 1
            self.column_count = max(4, int(np.ceil(self.obs_shape[0] / 4)))

    def __repr__(self) -> str:
        return f"GymGame({self.env_id!r})"

    def close(self) -> None:
        self.env.close()

    def get_initial_state(self) -> np.ndarray:
        obs, _ = self.env.reset()
        self._last_reward = 0.0
        self._last_done = False
        return self._postprocess_obs(obs)

    def get_next_state(self, state: np.ndarray, action: int, player: int = 1) -> np.ndarray:
        del player
        obs = state
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, _ = self.env.step(int(action))
            total_reward += float(reward)
            if terminated or truncated:
                break

        self._last_reward = total_reward
        self._last_done = terminated or truncated
        return self._postprocess_obs(obs, previous=state)

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        del state
        return np.ones(self.action_size, dtype=np.uint8)

    def get_value_and_terminated(self, state: np.ndarray, action: int | None) -> tuple[float, bool]:
        del state, action
        return self._last_reward, self._last_done

    def get_opponent(self, player: int) -> int:
        return player

    def get_opponent_value(self, value: float) -> float:
        return value

    def change_perspective(self, state: np.ndarray, player: int) -> np.ndarray:
        del player
        return state

    def get_encoded_state(self, state: np.ndarray) -> np.ndarray:
        if self.obs_kind == "vector":
            encoded = np.asarray(state, dtype=np.float32)
            if encoded.ndim == 1:
                encoded = encoded.reshape(1, -1)
            return encoded

        frames = np.asarray(state, dtype=np.float32)
        if frames.ndim == 2:
            frames = frames[np.newaxis, ...]
        return frames / 255.0

    def _postprocess_obs(self, obs: np.ndarray, previous: np.ndarray | None = None) -> np.ndarray:
        del previous
        return np.asarray(obs, dtype=np.float32)


class AtariGym(GymGame):
    """Atari with grayscale resize and frame stacking (MuZero-style observations)."""

    def __init__(
        self,
        env_id: str = "ALE/Pong-v5",
        *,
        frame_stack: int = 4,
        frame_skip: int = 4,
        screen_size: int = 84,
        max_episode_steps: int = 108_000,
        render_mode: str | None = None,
    ):
        self.screen_size = screen_size
        self._frames: deque[np.ndarray] = deque(maxlen=frame_stack)
        super().__init__(
            env_id,
            frame_stack=frame_stack,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
        )
        self.obs_kind = "image"
        self.row_count = 6
        self.column_count = 6

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise ImportError("opencv-python is required for Atari preprocessing") from exc

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(frame, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    def _postprocess_obs(self, obs: np.ndarray, previous: np.ndarray | None = None) -> np.ndarray:
        del previous
        frame = self._preprocess_frame(np.asarray(obs))
        if not self._frames:
            for _ in range(self.frame_stack):
                self._frames.append(frame)
        else:
            self._frames.append(frame)
        return np.stack(list(self._frames), axis=0)

    def get_initial_state(self) -> np.ndarray:
        self._frames.clear()
        return super().get_initial_state()
