"""Tests for Gym / MuZero integration (CartPole — no Atari ROMs)."""

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

from alphazero.gym_env import GymGame  # noqa: E402
from alphazero.muzero_atari import MuZeroVectorNetwork  # noqa: E402
from alphazero.muzero_gym import MuZeroGym, MuZeroGymMCTS  # noqa: E402


@pytest.fixture
def cartpole():
    game = GymGame("CartPole-v1", max_episode_steps=50)
    yield game
    game.close()


def test_gym_game_episode(cartpole):
    state = cartpole.get_initial_state()
    assert state.shape == (4,)
    for _ in range(10):
        action = cartpole.env.action_space.sample()
        state = cartpole.get_next_state(state, action)
        _, done = cartpole.get_value_and_terminated(state, action)
        if done:
            break


def test_muzero_vector_forward(cartpole):
    obs_dim = cartpole.get_encoded_state(cartpole.get_initial_state()).reshape(-1).shape[0]
    model = MuZeroVectorNetwork(cartpole, obs_dim=obs_dim, hidden_dim=32)
    x = torch.tensor(cartpole.get_encoded_state(cartpole.get_initial_state()), dtype=torch.float32).unsqueeze(0)
    hidden, policy, value = model.initial_inference(x)
    assert hidden.ndim == 4
    assert policy.shape[-1] == cartpole.action_size
    assert value.shape == (1, 1)


def test_muzero_gym_mcts(cartpole):
    obs_dim = cartpole.get_encoded_state(cartpole.get_initial_state()).reshape(-1).shape[0]
    model = MuZeroVectorNetwork(cartpole, obs_dim=obs_dim, hidden_dim=32)
    model.eval()
    args = {
        "C": 1.25,
        "gamma": 0.99,
        "num_searches": 5,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3,
    }
    mcts = MuZeroGymMCTS(cartpole, model, args)
    probs = mcts.search(cartpole.get_initial_state())
    assert probs.shape == (cartpole.action_size,)
    assert np.isclose(probs.sum(), 1.0)


def test_muzero_gym_self_play(cartpole):
    obs_dim = cartpole.get_encoded_state(cartpole.get_initial_state()).reshape(-1).shape[0]
    model = MuZeroVectorNetwork(cartpole, obs_dim=obs_dim, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    args = {
        "C": 1.25,
        "gamma": 0.99,
        "num_searches": 5,
        "num_iterations": 1,
        "num_selfPlay_iterations": 2,
        "num_epochs": 1,
        "batch_size": 2,
        "num_unroll_steps": 3,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3,
    }
    agent = MuZeroGym(model, optimizer, cartpole, args)
    model.eval()
    game = agent.self_play()
    assert len(game["observations"]) > 0
    assert len(game["policies"]) == len(game["observations"])
