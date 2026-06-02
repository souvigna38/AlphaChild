"""Gate C1-L08 — Connect Four."""

from alphazero.games import ConnectFour


def test_connectfour_init():
    g = ConnectFour()
    assert g.action_size == 7
    state = g.get_initial_state()
    assert g.get_valid_moves(state).sum() == 7
