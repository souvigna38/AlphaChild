# AGENTS.md

## Cursor Cloud specific instructions

This is a pure-Python Jupyter notebook project implementing AlphaZero for board games (TicTacToe, Connect Four). There are no traditional services to start — all code lives in 10 numbered `.ipynb` notebooks.

### Dependencies

Installed via pip: `numpy`, `torch` (CPU), `matplotlib`, `tqdm`, `ipykernel`, `jupyter`. No `requirements.txt` exists; the update script handles installation.

### Running notebooks

- All notebooks reference a kernel named `myenv` (the original author's environment). Override with `--ExecutePreprocessor.kernel_name=python3` when using `jupyter nbconvert --execute`.
- Several notebooks (1, 2, 3, 4, 5) contain interactive `input()` calls for playing games, so they cannot be fully executed non-interactively. To test non-interactive cells, extract code into a standalone `.py` script or use `nbclient` programmatically on selected cells.
- Pre-trained model weights (`model_2.pt`, `model_7_ConnectFour.pt`) are committed to the repo and can be loaded with `torch.load(..., map_location="cpu", weights_only=True)`.

### Testing

There is no test suite or linter configured. Validation is done by running notebook cells or extracting code into scripts. A quick smoke test: import all key libraries and load a pre-trained model to verify the environment works.

### PATH note

Pip installs scripts to `~/.local/bin`. Ensure `PATH` includes this directory (e.g., `export PATH="$HOME/.local/bin:$PATH"`) when running `jupyter` commands.
