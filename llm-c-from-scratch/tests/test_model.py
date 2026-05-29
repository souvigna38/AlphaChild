import torch

from llmc.data import CharTokenizer, load_text
from llmc.model import GPT, GPTConfig
from llmc.sample import generate

DATA = __import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "tiny_shakespeare.txt"


def test_gpt_forward():
    text = load_text(DATA)[:2000]
    tok = CharTokenizer.from_text(text)
    config = GPTConfig.tiny(tok.vocab_size, block_size=32)
    model = GPT(config)
    x = torch.randint(0, tok.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, tok.vocab_size)
    assert loss is not None


def test_generate():
    text = load_text(DATA)[:1000]
    tok = CharTokenizer.from_text(text)
    config = GPTConfig.tiny(tok.vocab_size, block_size=32)
    model = GPT(config)
    ctx = torch.tensor([[0, 1, 2, 3]])
    out = generate(model, ctx, max_new_tokens=5)
    assert out.shape[1] == ctx.shape[1] + 5
