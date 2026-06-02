# AlphaZeroFromScratch
Out of the box implementation based on the code of the tutorial: [AlphaZero](https://github.com/foersterrobert/AlphaZero)


![tictactoe](https://raw.githubusercontent.com/foersterrobert/AlphaZero/master/assets/tictactoe.gif)
![connectfour](https://raw.githubusercontent.com/foersterrobert/AlphaZero/master/assets/connectfour.gif)

### Local setup (Cursor agents & developers)

See **[AGENTS.md](AGENTS.md)** for step-by-step instructions: clone, virtualenv, `pip install -e ".[dev,atari]"`, run tests, and start Jupyter.

**Online course:** **[docs/ONLINE_COURSE.md](docs/ONLINE_COURSE.md)** — beginner curriculum for Track A (notebooks 1–16) and Track B (`llm-c-from-scratch/` 1–18).

Quick start:

```bash
git clone https://github.com/souvigna38/AlphaChild.git && cd AlphaChild
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,atari]"
python3 -m pytest tests/ -v
jupyter lab
```

### MuZero + Gym / Atari (notebooks 11–16)

After the AlphaZero series, notebooks **11–13** introduce MuZero on board games. **14–16** extend the same style to [Gymnasium](https://gymnasium.farama.org/) and Atari (frame stacking, visual encoder, single-agent MCTS).

### Some Helpful Resources
* AlphaZero-Paper: https://arxiv.org/pdf/1712.01815.pdf
* MuZero-Paper: https://arxiv.org/pdf/1911.08265.pdf
* Paper-Walkthrough: https://youtu.be/0slFo1rV0EM
* MCTS-Explained: https://youtu.be/UXW2yZndl7U
* AlphaZero-Explained: https://youtu.be/62nq4Zsn8vc

❤️
