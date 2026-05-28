import torch

from llmc.data import CharTokenizer, load_text, train_val_split
from llmc.deepseek_v2 import DeepSeekV2, DeepSeekV2Config

DATA = __import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "tiny_shakespeare.txt"


def test_mla_forward_and_cache():
    text = load_text(DATA)[:2000]
    tok = CharTokenizer.from_text(text)
    cfg = DeepSeekV2Config.tiny(tok.vocab_size, 32)
    model = DeepSeekV2(cfg)
    x = torch.randint(0, tok.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, tok.vocab_size)
    assert model.blocks[0].attn.last_c_kv is not None
    assert model.blocks[0].attn.last_c_kv.shape[-1] == cfg.kv_lora_rank
    assert model.kv_cache_bytes_per_token() < model.mha_kv_cache_bytes_per_token()


def test_train_short():
    text = load_text(DATA)[:4000]
    train_text, val_text = train_val_split(text)
    tok = CharTokenizer.from_text(text)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    cfg = DeepSeekV2Config.tiny(tok.vocab_size, 32)
    model = DeepSeekV2(cfg)
    from llmc.train import Trainer, TrainConfig

    trainer = Trainer(
        model,
        train_ids,
        val_ids,
        TrainConfig(max_steps=5, batch_size=4, eval_interval=5, eval_iters=2),
    )
    trainer.train()
