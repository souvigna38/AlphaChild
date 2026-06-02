from alphazero.notebook_utils import checkpoint_path, find_repo_root


def test_find_repo_root():
    root = find_repo_root()
    assert (root / "alphazero").is_dir()
    assert (root / "1.TicTacToe.ipynb").is_file()


def test_checkpoint_path():
    root = find_repo_root()
    assert checkpoint_path(root, "model_2.pt").name == "model_2.pt"
