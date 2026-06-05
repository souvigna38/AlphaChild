from pathlib import Path

from llmc.data import CharTokenizer, load_text, train_val_split

DATA = Path(__file__).resolve().parent.parent / "data" / "tiny_shakespeare.txt"


def test_load_text():
    text = load_text(DATA)
    assert len(text) > 10_000


def test_tokenizer_roundtrip():
    text = load_text(DATA)[:500]
    tok = CharTokenizer.from_text(text)
    sample = text[:20]
    encoded = tok.encode(sample)
    assert tok.decode(encoded) == sample


def test_train_val_split():
    text = load_text(DATA)
    train, val = train_val_split(text, 0.9)
    assert len(train) + len(val) == len(text)
