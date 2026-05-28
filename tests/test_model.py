import pytest
import torch

from alphazero.games import ConnectFour, TicTacToe
from alphazero.model import ResBlock, ResNet


class TestResBlock:
    def test_output_shape_matches_input(self):
        block = ResBlock(64)
        x = torch.randn(1, 64, 3, 3)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        block = ResBlock(64)
        block.eval()
        x = torch.randn(1, 64, 3, 3)
        out = block(x)
        assert out.shape == x.shape
        assert not torch.equal(out, x)


class TestResNetTicTacToe:
    @pytest.fixture
    def game(self):
        return TicTacToe()

    @pytest.fixture
    def model(self, game):
        m = ResNet(game, 4, 64)
        m.eval()
        return m

    def test_output_shapes(self, game, model):
        x = torch.randn(1, 3, 3, 3)
        policy, value = model(x)
        assert policy.shape == (1, 9)
        assert value.shape == (1, 1)

    def test_value_in_range(self, game, model):
        x = torch.randn(1, 3, 3, 3)
        _, value = model(x)
        assert -1 <= value.item() <= 1

    def test_batch_inference(self, game, model):
        x = torch.randn(4, 3, 3, 3)
        policy, value = model(x)
        assert policy.shape == (4, 9)
        assert value.shape == (4, 1)

    def test_with_encoded_state(self, game, model):
        state = game.get_initial_state()
        state = game.get_next_state(state, 4, 1)
        encoded = game.get_encoded_state(state)
        tensor = torch.tensor(encoded).unsqueeze(0)
        policy, value = model(tensor)
        assert policy.shape == (1, 9)
        assert value.shape == (1, 1)

    def test_load_pretrained_weights(self, game):
        model = ResNet(game, 4, 64)
        model.load_state_dict(torch.load("model_2.pt", map_location="cpu", weights_only=True))
        model.eval()
        state = game.get_initial_state()
        state = game.get_next_state(state, 4, 1)
        encoded = game.get_encoded_state(state)
        tensor = torch.tensor(encoded).unsqueeze(0)
        with torch.no_grad():
            policy, value = model(tensor)
        assert policy.shape == (1, 9)
        assert -1 <= value.item() <= 1


class TestResNetConnectFour:
    @pytest.fixture
    def game(self):
        return ConnectFour()

    def test_output_shapes(self, game):
        model = ResNet(game, 4, 64)
        model.eval()
        x = torch.randn(1, 3, 6, 7)
        policy, value = model(x)
        assert policy.shape == (1, 7)
        assert value.shape == (1, 1)

    def test_device_parameter(self, game):
        model = ResNet(game, 4, 64, device=torch.device("cpu"))
        assert model.device == torch.device("cpu")
