"""Gate C1-L03 — policy/value net."""

import torch

from alphazero.games import TicTacToe
from alphazero.model import ResNet


def test_model_forward():
    game = TicTacToe()
    net = ResNet(game, num_resBlocks=2, num_hidden=32)
    encoded = game.get_encoded_state(game.get_initial_state())
    x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
    pi, v = net(x)
    assert pi.shape[-1] == game.getActionSize()
    assert v.shape == (1, 1)
