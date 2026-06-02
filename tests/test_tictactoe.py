import numpy as np
import pytest

from alphazero.games import TicTacToe


@pytest.fixture
def game():
    return TicTacToe()


class TestTicTacToeInit:
    def test_dimensions(self, game):
        assert game.row_count == 3
        assert game.column_count == 3
        assert game.action_size == 9

    def test_repr(self, game):
        assert repr(game) == "TicTacToe"


class TestInitialState:
    def test_shape(self, game):
        state = game.get_initial_state()
        assert state.shape == (3, 3)

    def test_all_zeros(self, game):
        state = game.get_initial_state()
        assert np.all(state == 0)


class TestGetNextState:
    def test_place_piece(self, game):
        state = game.get_initial_state()
        state = game.get_next_state(state, 0, 1)
        assert state[0, 0] == 1

    def test_place_opponent_piece(self, game):
        state = game.get_initial_state()
        state = game.get_next_state(state, 4, -1)
        assert state[1, 1] == -1

    def test_action_to_position_mapping(self, game):
        state = game.get_initial_state()
        state = game.get_next_state(state, 8, 1)
        assert state[2, 2] == 1


class TestGetValidMoves:
    def test_initial_all_valid(self, game):
        state = game.get_initial_state()
        valid = game.get_valid_moves(state)
        assert np.all(valid == 1)
        assert valid.shape == (9,)

    def test_occupied_square_invalid(self, game):
        state = game.get_initial_state()
        state = game.get_next_state(state, 4, 1)
        valid = game.get_valid_moves(state)
        assert valid[4] == 0
        assert np.sum(valid) == 8


class TestCheckWin:
    def test_no_win_on_none_action(self, game):
        state = game.get_initial_state()
        assert not game.check_win(state, None)

    def test_row_win(self, game):
        state = game.get_initial_state()
        state = game.get_next_state(state, 0, 1)
        state = game.get_next_state(state, 1, 1)
        state = game.get_next_state(state, 2, 1)
        assert game.check_win(state, 2)

    def test_column_win(self, game):
        state = game.get_initial_state()
        state = game.get_next_state(state, 0, 1)
        state = game.get_next_state(state, 3, 1)
        state = game.get_next_state(state, 6, 1)
        assert game.check_win(state, 6)

    def test_diagonal_win(self, game):
        state = game.get_initial_state()
        state = game.get_next_state(state, 0, 1)
        state = game.get_next_state(state, 4, 1)
        state = game.get_next_state(state, 8, 1)
        assert game.check_win(state, 8)

    def test_anti_diagonal_win(self, game):
        state = game.get_initial_state()
        state = game.get_next_state(state, 2, -1)
        state = game.get_next_state(state, 4, -1)
        state = game.get_next_state(state, 6, -1)
        assert game.check_win(state, 6)

    def test_no_win_yet(self, game):
        state = game.get_initial_state()
        state = game.get_next_state(state, 0, 1)
        state = game.get_next_state(state, 1, -1)
        assert not game.check_win(state, 1)


class TestGetValueAndTerminated:
    def test_win_returns_one_true(self, game):
        state = game.get_initial_state()
        for a in [0, 1, 2]:
            state = game.get_next_state(state, a, 1)
        value, terminated = game.get_value_and_terminated(state, 2)
        assert value == 1
        assert terminated

    def test_draw(self, game):
        state = game.get_initial_state()
        moves = [(0, 1), (1, -1), (2, 1), (4, -1), (3, 1), (5, -1), (7, 1), (6, -1), (8, 1)]
        for action, player in moves:
            state = game.get_next_state(state, action, player)
        value, terminated = game.get_value_and_terminated(state, 8)
        assert value == 0
        assert terminated

    def test_game_in_progress(self, game):
        state = game.get_initial_state()
        state = game.get_next_state(state, 4, 1)
        value, terminated = game.get_value_and_terminated(state, 4)
        assert value == 0
        assert not terminated


class TestOpponent:
    def test_get_opponent(self, game):
        assert game.get_opponent(1) == -1
        assert game.get_opponent(-1) == 1

    def test_get_opponent_value(self, game):
        assert game.get_opponent_value(1) == -1
        assert game.get_opponent_value(0) == 0


class TestPerspective:
    def test_change_perspective_player1(self, game):
        state = game.get_initial_state()
        state = game.get_next_state(state, 0, 1)
        state = game.get_next_state(state, 4, -1)
        flipped = game.change_perspective(state.copy(), -1)
        assert flipped[0, 0] == -1
        assert flipped[1, 1] == 1


class TestEncodedState:
    def test_shape(self, game):
        state = game.get_initial_state()
        encoded = game.get_encoded_state(state)
        assert encoded.shape == (3, 3, 3)

    def test_encoding_values(self, game):
        state = game.get_initial_state()
        state = game.get_next_state(state, 0, 1)
        state = game.get_next_state(state, 4, -1)
        encoded = game.get_encoded_state(state)
        assert encoded[0, 1, 1] == 1  # channel 0 = player -1 positions
        assert encoded[2, 0, 0] == 1  # channel 2 = player 1 positions
        assert encoded[1, 0, 1] == 1  # channel 1 = empty positions

    def test_dtype_is_float32(self, game):
        state = game.get_initial_state()
        encoded = game.get_encoded_state(state)
        assert encoded.dtype == np.float32
