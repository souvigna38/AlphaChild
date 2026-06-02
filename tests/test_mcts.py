import numpy as np
import pytest
import torch

from alphazero.games import TicTacToe
from alphazero.mcts import MCTS, Node
from alphazero.model import ResNet


@pytest.fixture
def game():
    return TicTacToe()


@pytest.fixture
def model(game):
    m = ResNet(game, 4, 64)
    m.eval()
    return m


@pytest.fixture
def mcts_args():
    return {
        'C': 2,
        'num_searches': 50,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,
    }


class TestNode:
    def test_initial_state(self, game, mcts_args):
        state = game.get_initial_state()
        node = Node(game, mcts_args, state)
        assert node.visit_count == 0
        assert node.value_sum == 0
        assert len(node.children) == 0
        assert node.parent is None

    def test_not_expanded_initially(self, game, mcts_args):
        state = game.get_initial_state()
        node = Node(game, mcts_args, state)
        assert node.is_fully_expanded() is False

    def test_expand_creates_children(self, game, mcts_args):
        state = game.get_initial_state()
        node = Node(game, mcts_args, state)
        policy = np.ones(game.action_size) / game.action_size
        node.expand(policy)
        assert len(node.children) == game.action_size
        assert node.is_fully_expanded() is True

    def test_backpropagate(self, game, mcts_args):
        state = game.get_initial_state()
        parent = Node(game, mcts_args, state)
        child_state = state.copy()
        child = Node(game, mcts_args, child_state, parent=parent, action_taken=0)
        parent.children.append(child)

        child.backpropagate(0.5)
        assert child.visit_count == 1
        assert child.value_sum == 0.5
        assert parent.visit_count == 1
        assert parent.value_sum == -0.5

    def test_select_returns_child(self, game, mcts_args):
        state = game.get_initial_state()
        node = Node(game, mcts_args, state, visit_count=1)
        policy = np.ones(game.action_size) / game.action_size
        node.expand(policy)
        for child in node.children:
            child.visit_count = 1
            child.value_sum = 0.5
        selected = node.select()
        assert selected is not None
        assert selected in node.children


class TestMCTS:
    def test_search_returns_valid_probs(self, game, model, mcts_args):
        np.random.seed(42)
        mcts = MCTS(game, mcts_args, model)
        state = game.get_initial_state()
        probs = mcts.search(state)
        assert probs.shape == (9,)
        assert abs(np.sum(probs) - 1.0) < 1e-6
        assert np.all(probs >= 0)

    def test_search_respects_valid_moves(self, game, model, mcts_args):
        np.random.seed(42)
        mcts = MCTS(game, mcts_args, model)
        state = game.get_initial_state()
        state = game.get_next_state(state, 4, 1)
        neutral = game.change_perspective(state, -1)
        probs = mcts.search(neutral)
        assert probs[4] == 0  # center is occupied

    def test_pretrained_model_picks_good_moves(self, game, mcts_args):
        np.random.seed(42)
        model = ResNet(game, 4, 64)
        model.load_state_dict(torch.load("model_2.pt", map_location="cpu", weights_only=True))
        model.eval()
        mcts_args['num_searches'] = 100
        mcts = MCTS(game, mcts_args, model)

        state = game.get_initial_state()
        state = game.get_next_state(state, 0, 1)
        state = game.get_next_state(state, 4, -1)
        state = game.get_next_state(state, 2, 1)
        neutral = game.change_perspective(state, -1)
        probs = mcts.search(neutral)
        assert probs.shape == (9,)
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_full_game_completes(self, game, model, mcts_args):
        np.random.seed(42)
        mcts = MCTS(game, mcts_args, model)
        state = game.get_initial_state()
        player = 1
        for _ in range(9):
            neutral = game.change_perspective(state.copy(), player)
            probs = mcts.search(neutral)
            action = np.argmax(probs)
            state = game.get_next_state(state, action, player)
            value, terminated = game.get_value_and_terminated(state, action)
            if terminated:
                break
            player = game.get_opponent(player)
        assert terminated
