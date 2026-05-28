import torch

from llmc.data import CharTokenizer, load_text, train_val_split
from llmc.model import GPT, GPTConfig
from llmc.train import Trainer, TrainConfig

DATA = __import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "tiny_shakespeare.txt"


def test_trainer_short_run():
    text = load_text(DATA)[:5000]
    train_text, val_text = train_val_split(text)
    tok = CharTokenizer.from_text(text)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    model = GPT(GPTConfig.tiny(tok.vocab_size, block_size=32))
    trainer = Trainer(
        model,
        train_ids,
        val_ids,
        TrainConfig(max_steps=10, batch_size=4, eval_interval=5, eval_iters=2),
    )
    history = trainer.train()
    assert len(history) >= 1
    assert history[-1]["val"] > 0
