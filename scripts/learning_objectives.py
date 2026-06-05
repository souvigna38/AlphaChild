"""Learning objectives for all course notebooks (Track A + Track B)."""

from __future__ import annotations

# Track A — AlphaZero / MuZero (repo root)
ALPHA_OBJECTIVES: dict[str, list[str]] = {
    "1.TicTacToe.ipynb": [
        "Represent a game state as a NumPy array.",
        "List legal moves and apply an action to get the next state.",
        "Detect wins, draws, and whose turn it is.",
        "Explain reward and termination for reinforcement learning.",
    ],
    "2.MCTS.ipynb": [
        "Implement selection, expansion, simulation, and backpropagation.",
        "Use UCB1 to balance exploration vs exploitation.",
        "Run MCTS to choose stronger moves than random play.",
        "Describe how search improves without any neural network.",
    ],
    "3.Model.ipynb": [
        "Encode a board as a tensor input to a neural network.",
        "Build a ResNet-style policy head (move probabilities).",
        "Build a value head (expected outcome from a position).",
        "Connect the model output to MCTS priors and leaf evaluation.",
    ],
    "4.AlphaMCTS.ipynb": [
        "Replace rollout simulation with neural network value estimates.",
        "Use policy logits as MCTS priors (PUCT).",
        "Play a full game with AlphaZero-style search.",
        "Contrast random rollouts (nb 2) with learned evaluation (this notebook).",
    ],
    "5.AlphaSelfPlay.ipynb": [
        "Collect self-play games into a replay buffer.",
        "Train the network on (state, policy target, value target) tuples.",
        "Run the complete generate-data → train loop once.",
        "Explain why self-play needs no human game labels.",
    ],
    "6.AlphaTrain.ipynb": [
        "Run multiple training iterations on Tic-Tac-Toe.",
        "Load a saved checkpoint when available.",
        "Track how loss and playing strength change over iterations.",
        "Set `RUN_TRAIN=True` when ready for a full training run.",
    ],
    "7.AlphaTweaks.ipynb": [
        "Apply learning-rate decay during training.",
        "Use temperature and Dirichlet noise in self-play.",
        "Explain how each tweak affects exploration and stability.",
        "Compare default vs tuned hyperparameters.",
    ],
    "8.ConnectFour.ipynb": [
        "Adapt the game class to a larger board (6×7).",
        "Reuse the same AlphaZero pipeline on a harder game.",
        "Recognize when more search and training are needed.",
        "Train or evaluate on Connect Four with `RUN_TRAIN`.",
    ],
    "9.AlphaParallel.ipynb": [
        "Collect self-play data from multiple games in parallel.",
        "Understand throughput vs single-game self-play.",
        "Load Connect Four checkpoints when shipped in the repo.",
        "Identify when parallelization helps wall-clock time.",
    ],
    "10.Eval.ipynb": [
        "Measure agent strength with repeated matches.",
        "Compare models on Tic-Tac-Toe and Connect Four.",
        "Optionally use the Kaggle Connect Four environment.",
        "Interpret win rates as an evaluation metric.",
    ],
    "11.MuZeroModel.ipynb": [
        "Define representation, dynamics, and prediction networks.",
        "Explain how MuZero plans without a known game simulator.",
        "Map board states to latent vectors for planning.",
        "Contrast MuZero model with AlphaZero policy/value-only net.",
    ],
    "12.MuZeroMCTS.ipynb": [
        "Run MCTS in learned latent space (not raw board).",
        "Use the dynamics network to imagine future states.",
        "Combine policy, value, and reward predictions in search.",
        "Play using MuZero MCTS on a board game.",
    ],
    "13.MuZeroTrain.ipynb": [
        "Run the MuZero training loop (self-play + train).",
        "Understand reanalysis and target construction at a high level.",
        "Train on board games with `RUN_TRAIN=True`.",
        "Relate this loop to AlphaZero training (nb 5–6).",
    ],
    "14.MuZeroGym.ipynb": [
        "Wrap a Gymnasium environment for MuZero.",
        "Handle vector observations (e.g. CartPole).",
        "See how the same algorithm generalizes beyond board games.",
        "Run the CartPole demo cell successfully.",
    ],
    "15.MuZeroAtariModel.ipynb": [
        "Stack frames for visual Atari inputs.",
        "Build a convolutional encoder for pixel observations.",
        "Understand why Atari is optional (ROMs, memory).",
        "Compare visual encoder design to board encoders.",
    ],
    "16.MuZeroAtariTrain.ipynb": [
        "Train MuZero on CartPole with small iteration counts.",
        "Attempt Atari training when ROMs and deps are available.",
        "Use `RUN_TRAIN` for longer runs.",
        "Interpret training logs for classic control vs Atari.",
    ],
}

# Track B — LLM / llm.c (llm-c-from-scratch/)
LLM_OBJECTIVES: dict[str, list[str]] = {
    "1.Tokens.ipynb": [
        "Load raw text and inspect corpus size.",
        "Build a character-level vocabulary (encode / decode).",
        "Convert text to integer token IDs and PyTorch tensors.",
        "Relate character tokens to llm.c `.bin` token files.",
    ],
    "2.Bigram.ipynb": [
        "Count pairwise character frequencies (bigram table).",
        "Apply smoothing so every context has a valid distribution.",
        "Sample text from the bigram model.",
        "Recognize this as the simplest language model baseline.",
    ],
    "3.Embeddings.ipynb": [
        "Map token IDs to vectors with `nn.Embedding` (wte).",
        "Add position embeddings (wpe) like GPT-2.",
        "Combine token + position into a hidden state.",
        "Connect embeddings to the transformer input in llm.c.",
    ],
    "4.Attention.ipynb": [
        "Implement scaled dot-product attention on synthetic data.",
        "Apply a causal mask so tokens only see the past.",
        "Split into multi-head attention (GPT-2 layout).",
        "Explain Q, K, V shapes and output dimensions.",
    ],
    "5.GPTBlock.ipynb": [
        "Run one `Block`: LayerNorm → attention → residual → MLP → residual.",
        "Forward a batch through `llmc.model.Block`.",
        "Match block structure to one layer in `train_gpt2.c`.",
        "Verify input and output shapes are unchanged.",
    ],
    "6.GPT.ipynb": [
        "Stack blocks into a full `GPT` module.",
        "Run forward pass and compute cross-entropy loss.",
        "Count parameters for the tiny Shakespeare model.",
        "Relate `GPT` layout to notebook 5 and llm.c model struct.",
    ],
    "7.BatchAndLoss.ipynb": [
        "Sample random contiguous batches from tokenized Shakespeare.",
        "Move data to CPU or GPU and run one training-style forward.",
        "Read a single batch loss value.",
        "Explain why batching is required for efficient training.",
    ],
    "8.Train.ipynb": [
        "Configure `Trainer` with steps, batch size, and learning rate.",
        "Run training when `RUN_TRAIN=True` and read train/val loss.",
        "Save `checkpoints/tiny_gpt.pt` for notebook 9.",
        "Map the loop to `train_gpt2.py` / llm.c main training loop.",
    ],
    "9.Sample.ipynb": [
        "Load a checkpoint from notebook 8 when available.",
        "Generate text with `generate()` (temperature, top-k).",
        "Compare untrained vs trained sample quality.",
        "Relate sampling to inference in llm.c / nanoGPT.",
    ],
    "10.GPT2AndLlmc.ipynb": [
        "List GPT-2 124M hyperparameters (layers, heads, embd).",
        "Compare tiny model param count vs GPT-2 small.",
        "Map each notebook (1–10) to llm.c source files.",
        "Know when to set `RUN_GPT2=True` (RAM-heavy allocation only).",
    ],
    "11.DeepSeekPath.ipynb": [
        "Explain why DeepSeek-V2 follows GPT-2 in this curriculum.",
        "Compare GPT-2 block vs DeepSeek-V2 block (MLA + MoE).",
        "Estimate KV-cache size: MHA vs MLA.",
        "Locate the V2 C port under `c/deepseek_v2/`.",
    ],
    "12.MLA.ipynb": [
        "Forward pass through `MultiHeadLatentAttention`.",
        "Identify `c_kv` as the inference cache (not full K, V).",
        "Compare MLA weight shapes to GPT-2 `c_attn`.",
        "Quantify cache bytes per token vs hypothetical MHA.",
    ],
    "13.DeepSeekMoE.ipynb": [
        "Run one SwiGLU expert and full `DeepSeekMoE` block.",
        "Trace router softmax, top-k experts, and renormalization.",
        "Compare dense GPT MLP params vs MoE params on one block.",
        "Relate router logic to `c/deepseek_v2/moe.c`.",
    ],
    "14.DeepSeekV2Model.ipynb": [
        "Build full `DeepSeekV2` and run forward + loss.",
        "Inspect block children: RMSNorm, MLA, MoE.",
        "Trace shapes through one block manually.",
        "Connect to notebook 6 (full GPT) at the same role.",
    ],
    "15.TrainDeepSeekV2.ipynb": [
        "Set up `Trainer` for DeepSeek-V2 on tiny Shakespeare.",
        "Choose PyTorch training (`RUN_TRAIN`) or C head-only demo.",
        "Read device and parameter count before training.",
        "Compare training options to notebook 8 (GPT).",
    ],
    "16.SampleDeepSeekV2.ipynb": [
        "Generate text with untrained or trained V2 model.",
        "Export weights for C with `RUN_EXPORT=True` after training.",
        "Run C greedy sample when checkpoint exists.",
        "Explain train-in-PyTorch, sample-in-C workflow.",
    ],
    "17.Phase5CBackward.ipynb": [
        "Verify MLA backward with PyTorch autograd.",
        "List C files for backward: MLA, MoE, block, AdamW.",
        "Run `make test_v2` and `-train-1layer` from the terminal.",
        "Place Phase 5 C backward in the overall V2 roadmap.",
    ],
    "18.DeepSeekV4Path.ipynb": [
        "Configure `DeepSeekV4Config.tiny()` for fast demos.",
        "Run HashMoE, SwiGLU, and RoutedMoE forward passes.",
        "Build hash routing table and match `c/hash_moe.c`.",
        "Pass fast pytest checks and optional Dojo gate C2-L18.",
    ],
}


def format_objectives(objectives: list[str]) -> str:
    lines = ["**Learning objectives**", ""]
    for obj in objectives:
        lines.append(f"- {obj}")
    return "\n".join(lines)
