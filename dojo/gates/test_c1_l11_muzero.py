"""Gate C1-L11 — MuZero vector net (CartPole-sized)."""

import pytest
import torch

pytest.importorskip("gymnasium")

from alphazero.gym_env import GymGame
from alphazero.muzero_atari import MuZeroVectorNetwork


def test_muzero_net():
    game = GymGame("CartPole-v1", max_episode_steps=20)
    try:
        obs = game.get_encoded_state(game.get_initial_state()).reshape(-1)
        model = MuZeroVectorNetwork(game, obs_dim=obs.shape[0], hidden_dim=32)
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        hidden, policy, value = model.initial_inference(x)
        assert policy.shape[-1] == game.action_size
        assert value.shape == (1, 1)
        assert hidden.ndim == 4
    finally:
        game.close()
