"""Gate C2-L01 — tokenizer."""

from llmc.data import CharTokenizer, load_text
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "llm-c-from-scratch" / "data" / "tiny_shakespeare.txt"


def test_char_tokenizer():
    if not DATA.is_file():
        text = "abc"
    else:
        text = load_text(DATA)[:500]
    tok = CharTokenizer.from_text(text)
    ids = tok.encode("hello")
    assert tok.decode(ids) == "hello"
