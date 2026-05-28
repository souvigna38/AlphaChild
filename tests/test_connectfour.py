import numpy as np
import pytest

from alphazero.games import ConnectFour


@pytest.fixture
def game():
    return ConnectFour()


class TestConnectFourInit:
    def test_dimensions(self, game):
        assert game.row_count == 6
        assert game.column_count == 7
        assert game.action_size == 7
        assert game.in_a_row == 4

    def test_repr(self, game):
        assert repr(game) == "ConnectFour"


class TestInitialState:
    def test_shape(self, game):
        state = game.get_initial_state()
        assert state.shape == (6, 7)

    def test_all_zeros(self, game):
        state = game.get_initial_state()
        assert np.all(state == 0)


class TestGetNextState:
    def test_piece_falls_to_bottom(self, game):
        state = game.get_initial_state()
        state = game.get_next_state(state, 3, 1)
        assert state[5, 3] == 1
        assert np.sum(state != 0) == 1

    def test_pieces_stack(self, game):
        state = game.get_initial_state()
        state = game.get_next_state(state, 3, 1)
        state = game.get_next_state(state, 3, -1)
        assert state[5, 3] == 1
        assert state[4, 3] == -1


class TestGetValidMoves:
    def test_initial_all_valid(self, game):
        state = game.get_initial_state()
        valid = game.get_valid_moves(state)
        assert np.all(valid == 1)
        assert valid.shape == (7,)

    def test_full_column_invalid(self, game):
        state = game.get_initial_state()
        for i in range(6):
            state = game.get_next_state(state, 0, 1 if i % 2 == 0 else -1)
        valid = game.get_valid_moves(state)
        assert valid[0] == 0
        assert np.sum(valid) == 6


class TestCheckWin:
    def test_no_win_on_none_action(self, game):
        state = game.get_initial_state()
        assert not game.check_win(state, None)

    def test_vertical_win(self, game):
        state = game.get_initial_state()
        for _ in range(4):
            state = game.get_next_state(state, 0, 1)
        assert game.check_win(state, 0)

    def test_horizontal_win(self, game):
        state = game.get_initial_state()
        for col in range(4):
            state = game.get_next_state(state, col, 1)
        assert game.check_win(state, 3)

    def test_no_win_three_in_row(self, game):
        state = game.get_initial_state()
        for col in range(3):
            state = game.get_next_state(state, col, 1)
        assert not game.check_win(state, 2)


class TestGetValueAndTerminated:
    def test_win(self, game):
        state = game.get_initial_state()
        for col in range(4):
            state = game.get_next_state(state, col, 1)
        value, terminated = game.get_value_and_terminated(state, 3)
        assert value == 1
        assert terminated

    def test_game_in_progress(self, game):
        state = game.get_initial_state()
        state = game.get_next_state(state, 3, 1)
        value, terminated = game.get_value_and_terminated(state, 3)
        assert value == 0
        assert not terminated


class TestOpponent:
    def test_get_opponent(self, game):
        assert game.get_opponent(1) == -1
        assert game.get_opponent(-1) == 1


class TestEncodedState:
    def test_shape(self, game):
        state = game.get_initial_state()
        encoded = game.get_encoded_state(state)
        assert encoded.shape == (3, 6, 7)

    def test_dtype_is_float32(self, game):
        state = game.get_initial_state()
        encoded = game.get_encoded_state(state)
        assert encoded.dtype == np.float32
