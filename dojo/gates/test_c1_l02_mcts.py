"""Gate C1-L02 — MCTS."""

import numpy as np

from alphazero.games import TicTacToe
from alphazero.mcts import MCTS
from alphazero.model import ResNet


def test_mcts_runs():
    game = TicTacToe()
    model = ResNet(game, num_resBlocks=2, num_hidden=32)
    model.eval()
    args = {
        "C": 2,
        "num_searches": 25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3,
    }
    mcts = MCTS(game, args, model)
    state = game.get_initial_state()
    pi = mcts.search(state)
    assert pi.shape == (9,)
    assert abs(float(pi.sum()) - 1.0) < 1e-5
