# AGENTS.md

## Cursor Cloud specific instructions

This is a pure-Python Jupyter notebook project implementing AlphaZero for board games (TicTacToe, Connect Four) and MuZero for Gym / Atari. There are no traditional services to start — all code lives in numbered `.ipynb` notebooks, with reusable classes extracted into the `alphazero/` package.

### Project structure

- **Notebooks** (`1.TicTacToe.ipynb` through `10.Eval.ipynb`): AlphaZero tutorial progression
- **MuZero notebooks** (`11.MuZeroModel.ipynb`–`13.MuZeroTrain.ipynb`): Board-game MuZero
- **Gym / Atari** (`14.MuZeroGym.ipynb`–`16.MuZeroAtariTrain.ipynb`): Gymnasium wrapper, Atari model, training
- **`alphazero/`**: Extracted package (`games.py`, `model.py`, `mcts.py`, `gym_env.py`, `muzero_atari.py`, `muzero_gym.py`)
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

Tests cover TicTacToe logic, ConnectFour logic, ResNet model, MCTS search, MuZero Gym (CartPole), pre-trained model loading, and notebook cell syntax validation. Non-interactive notebook cells are also tested by execution.

### Gym / Atari optional dependencies

```
pip install -e ".[gym]"        # CartPole and vector envs
pip install -e ".[atari]"      # Atari ROMs (accept license on first import)
```

Notebook cells that import `ALE/` or `AtariGym` are skipped in automated notebook syntax tests when ROMs are unavailable.

### Notebook caveats

- All notebooks reference a kernel named `myenv` (the original author's environment). Override with `--ExecutePreprocessor.kernel_name=python3` when using `jupyter nbconvert --execute`.
- Every notebook has one interactive `input()` cell (human-vs-AI game loop) that cannot run non-interactively. The test suite skips these cells automatically.
- Training cells (`.learn()`) are also skipped in tests to keep them fast.
- Pre-trained model weights can be loaded with `torch.load(..., map_location="cpu", weights_only=True)`.

### PATH note

Pip installs scripts to `~/.local/bin`. Ensure `PATH` includes this directory (e.g., `export PATH="$HOME/.local/bin:$PATH"`) when running `jupyter`, `ruff`, or `pytest` commands.
