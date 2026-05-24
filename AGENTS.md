# AGENTS.md

## Cursor Cloud specific instructions

This is a pure-Python Jupyter notebook project implementing AlphaZero for board games (TicTacToe, Connect Four). There are no traditional services to start — all code lives in 10 numbered `.ipynb` notebooks, with reusable classes extracted into the `alphazero/` package.

### Project structure

- **Notebooks** (`1.TicTacToe.ipynb` through `10.Eval.ipynb`): Tutorial progression
- **`alphazero/`**: Extracted Python package with `games.py`, `model.py`, `mcts.py`
- **`tests/`**: pytest test suite covering games, model, MCTS, and notebook syntax
- **`*.pt` files**: Pre-trained model weights for TicTacToe and ConnectFour

### Linting

```
ruff check alphazero/ tests/
```

Config is in `pyproject.toml`. Rules: pycodestyle, pyflakes, isort, pyupgrade, flake8-bugbear.

### Testing

```
python3 -m pytest tests/ -v
```

71 tests covering TicTacToe logic, ConnectFour logic, ResNet model, MCTS search, pre-trained model loading, and notebook cell syntax validation. Non-interactive notebook cells are also tested by execution.

### Notebook caveats

- All notebooks reference a kernel named `myenv` (the original author's environment). Override with `--ExecutePreprocessor.kernel_name=python3` when using `jupyter nbconvert --execute`.
- Every notebook has one interactive `input()` cell (human-vs-AI game loop) that cannot run non-interactively. The test suite skips these cells automatically.
- Training cells (`.learn()`) are also skipped in tests to keep them fast.
- Pre-trained model weights can be loaded with `torch.load(..., map_location="cpu", weights_only=True)`.

### PATH note

Pip installs scripts to `~/.local/bin`. Ensure `PATH` includes this directory (e.g., `export PATH="$HOME/.local/bin:$PATH"`) when running `jupyter`, `ruff`, or `pytest` commands.
