from pathlib import Path

from llmc.notebook_utils import checkpoint_path, data_path, find_llm_root


def test_find_llm_root():
    root = find_llm_root()
    assert (root / "llmc" / "__init__.py").is_file()
    assert data_path(root).is_file()


def test_checkpoint_path():
    root = find_llm_root()
    assert checkpoint_path(root).name == "tiny_gpt.pt"
