"""Gate C1-L01 — TicTacToe rules."""

from alphazero.games import TicTacToe


def test_tictactoe_legal_moves():
    g = TicTacToe()
    state = g.get_initial_state()
    assert g.get_valid_moves(state).sum() == 9
    state = g.get_next_state(state, 4, player=1)
    assert g.get_valid_moves(state).sum() == 8
