#!/usr/bin/env python3
"""Prepare LLM notebooks 1–18 for online beginner course."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETUP = """\
# --- Setup: find repo root (llm-c-from-scratch or Cursor workbook) ---
import sys
from pathlib import Path


def find_llm_root() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "llmc" / "__init__.py").is_file():
            return base
        nested = base / "llm-c-from-scratch"
        if (nested / "llmc" / "__init__.py").is_file():
            return nested
    return Path.cwd()


ROOT = find_llm_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmc.notebook_utils import c_dir, checkpoint_path, data_path, v2_checkpoint_path

DATA = data_path(ROOT)
CHECKPOINT = checkpoint_path(ROOT)
C_DIR = c_dir(ROOT)
V2_CKPT = v2_checkpoint_path(ROOT)
print("ROOT", ROOT.resolve())
print("data", "OK" if DATA.is_file() else "missing")
"""

COURSE_FOOTER = """\
## Next steps

- Run cells top-to-bottom before moving to the next notebook.
- Stuck? Re-read the **Before** section and check `ROOT` / `data OK` in the setup cell.
- Full curriculum: `docs/ONLINE_COURSE.md` in the AlphaChild repo.
"""

KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"},
}


def _src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _set_src(cell: dict, text: str) -> None:
    cell["source"] = [ln + "\n" for ln in text.strip("\n").split("\n")]


def has_setup(nb: dict) -> bool:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            s = _src(cell)
            if "find_llm_root" in s or ("ROOT = find_llm_root()" in s):
                return True
            if "find_repo_root" in s and "RUN_TRAIN" in s:
                return True
    return False


def remove_stale_setup_cells(nb: dict) -> None:
    keep = []
    for cell in nb.get("cells", []):
        s = _src(cell)
        if cell.get("cell_type") == "code" and 'assert Path("data/tiny_shakespeare.txt")' in s:
            continue
        if cell.get("cell_type") == "code" and s.strip().startswith("# --- Setup (same style"):
            continue
        keep.append(cell)
    nb["cells"] = keep


def insert_setup(nb: dict, after: int = 0) -> None:
    if has_setup(nb):
        return
    nb["cells"].insert(after + 1, {"cell_type": "code", "metadata": {}, "outputs": [], "source": []})
    _set_src(nb["cells"][after + 1], SETUP)


def fix_paths(text: str) -> str:
    text = text.replace('load_text("data/tiny_shakespeare.txt")', "load_text(DATA)")
    text = text.replace("load_text('data/tiny_shakespeare.txt')", "load_text(DATA)")
    text = text.replace('Path("data/tiny_shakespeare.txt")', "DATA")
    text = text.replace("Path('data/tiny_shakespeare.txt')", "DATA")
    text = text.replace('open("data/tiny_shakespeare.txt"', "open(DATA")
    text = text.replace('cwd="c"', "cwd=str(C_DIR)")
    text = text.replace('cwd="c/"', "cwd=str(C_DIR)")
    text = re.sub(r'Path\("checkpoints/([^"]+)"\)', r'ROOT / "checkpoints" / "\1"', text)
    text = re.sub(
        r'assert Path\("data/tiny_shakespeare\.txt"\)\.exists\(\).*',
        'assert DATA.is_file(), f"Missing {DATA} — check setup cell ROOT"',
        text,
    )
    text = text.replace('Path("vendor/llm.c/train_gpt2.c")', 'ROOT / "vendor" / "llm.c" / "train_gpt2.c"')
    return text


def gate_subprocess_make(text: str) -> str:
    if "subprocess.run" in text and "make" in text and "RUN_C" not in text:
        return (
            "RUN_C = False  # set True to compile/run C smokes\n\n"
            + text.replace(
                "if shutil.which(\"make\"):",
                "if RUN_C and shutil.which(\"make\"):",
            ).replace(
                "if shutil.which('make'):",
                "if RUN_C and shutil.which('make'):",
            )
        )
    return text


def gate_export_v2(text: str) -> str:
    if "export_v2_tiny.py" in text and "RUN_EXPORT" not in text:
        return (
            "RUN_EXPORT = False  # set True after training in notebook 15\n\n"
            + text.replace(
                "subprocess.run([",
                "if RUN_EXPORT:\n    subprocess.run([",
            )
            + "\nelse:\n    print(\"Export skipped — set RUN_EXPORT=True after training\")"
        )
    return text


def patch_cell(cell: dict) -> None:
    if cell.get("cell_type") != "code":
        return
    src = fix_paths(_src(cell))
    src = gate_subprocess_make(src)
    src = gate_export_v2(src)
    _set_src(cell, src)
    cell["outputs"] = []
    cell["execution_count"] = None


def ensure_course_header(nb: dict, title: str, before: str, body: str, dojo: str = "") -> None:
    header = (
        f"# {title}\n\n"
        f"**Before:** {before}\n\n"
        f"**This notebook:** {body}\n\n"
        "**Online course:** run cells **top-to-bottom**. Setup cell must print `data OK`.\n\n"
    )
    if dojo:
        header += f"**Dojo (optional):** `{dojo}`\n"
    if nb["cells"] and nb["cells"][0].get("cell_type") == "markdown":
        existing = _src(nb["cells"][0])
        if "Online course" not in existing:
            _set_src(nb["cells"][0], header + "\n" + existing.lstrip("# ").split("\n", 1)[-1] if existing.startswith("#") else header + existing)
    else:
        nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": []})
        _set_src(nb["cells"][0], header)


def patch_notebook(path: Path, meta: dict | None = None) -> None:
    nb = json.loads(path.read_text())
    if meta:
        ensure_course_header(nb, **meta)
    remove_stale_setup_cells(nb)
    if not has_setup(nb):
        idx = 0
        if nb["cells"] and nb["cells"][0].get("cell_type") == "markdown":
            idx = 1
        nb["cells"].insert(idx, {"cell_type": "code", "metadata": {}, "outputs": [], "source": []})
        _set_src(nb["cells"][idx], SETUP)
    for cell in nb["cells"]:
        patch_cell(cell)
    nb["metadata"].update(KERNEL)
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 4
    path.write_text(json.dumps(nb, indent=2) + "\n")
    print("patched", path.name)


def patch_nb8_train_skip() -> None:
    path = ROOT / "8.Train.ipynb"
    nb = json.loads(path.read_text())
    for cell in nb["cells"]:
        src = _src(cell)
        if "history = trainer.train()" in src and "if RUN_TRAIN:" not in src:
            _set_src(
                cell,
                """\
if RUN_TRAIN:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    history = trainer.train()
    for row in history:
        print(f"step {row['step']:4d} | train {row['train']:.4f} | val {row['val']:.4f}")
    torch.save({"model": trainer.model.state_dict(), "config": config}, CHECKPOINT)
    print("saved", CHECKPOINT)
else:
    print("Training skipped — set RUN_TRAIN=True (trainer configured above with max_steps)")""",
            )
    path.write_text(json.dumps(nb, indent=2) + "\n")
    print("patched 8.Train train gate")


def patch_nb10_gpt2() -> None:
    path = ROOT / "10.GPT2AndLlmc.ipynb"
    nb = json.loads(path.read_text())
    for cell in nb["cells"]:
        src = _src(cell)
        if "GPTConfig.gpt2_small(50257" in src and "RUN_GPT2" not in src:
            _set_src(
                cell,
                "RUN_GPT2 = False  # True allocates ~124M params (~500MB RAM)\n\n"
                + src.replace(
                    "gpt2 = GPT(GPTConfig.gpt2_small(50257, 1024))",
                    "gpt2 = GPT(GPTConfig.gpt2_small(50257, 1024)) if RUN_GPT2 else None",
                ).replace(
                    'print(f"gpt2 params:   {gpt2.count_parameters():,}")',
                    'print(f"gpt2 params:   {gpt2.count_parameters():,}" if gpt2 else "gpt2 skipped — set RUN_GPT2=True")',
                ),
            )
    path.write_text(json.dumps(nb, indent=2) + "\n")
    print("patched 10.GPT2AndLlmc gate")


def patch_nb15_train() -> None:
    path = ROOT / "15.TrainDeepSeekV2.ipynb"
    nb = json.loads(path.read_text())
    for cell in nb["cells"]:
        src = _src(cell)
        if "Uncomment history = trainer.train()" in src:
            _set_src(
                cell,
                """\
RUN_TRAIN = False  # True → 200 steps on CPU (~minutes)

if RUN_TRAIN:
    history = trainer.train()
    print("last val loss:", history[-1]["val"])
else:
    print("PyTorch training skipped — set RUN_TRAIN=True (or use C track below)")""",
            )
    path.write_text(json.dumps(nb, indent=2) + "\n")
    print("patched 15.TrainDeepSeekV2")


META = {
    "11.DeepSeekPath.ipynb": {
        "title": "11 — From GPT-2 to DeepSeek (roadmap)",
        "before": "notebooks **1–10** (GPT / llm.c path).",
        "body": "why we teach **DeepSeek-V2** after GPT-2, not “V1”; map to `c/deepseek_v2/`.",
    },
    "12.MLA.ipynb": {
        "title": "12 — Multi-head Latent Attention (MLA)",
        "before": "notebook **11**.",
        "body": "V2 attention — smaller KV cache than GPT-2 MHA.",
        "dojo": "dojo-grade --lesson C2-L12",
    },
    "13.DeepSeekMoE.ipynb": {
        "title": "13 — DeepSeekMoE",
        "before": "notebook **12** (MLA).",
        "body": "sparse FFN — router + top-k experts + shared expert.",
        "dojo": "dojo-grade --lesson C2-L13",
    },
    "14.DeepSeekV2Model.ipynb": {
        "title": "14 — Full DeepSeek-V2 tiny model",
        "before": "notebooks **12–13**.",
        "body": "stack MLA + MoE into `DeepSeekV2` (like notebook 6 for GPT).",
    },
    "15.TrainDeepSeekV2.ipynb": {
        "title": "15 — Train DeepSeek-V2",
        "before": "notebook **14**.",
        "body": "PyTorch `Trainer` or optional C head-only demo.",
        "dojo": "dojo-grade --lesson C2-L15",
    },
    "16.SampleDeepSeekV2.ipynb": {
        "title": "16 — Sample + export to C",
        "before": "notebook **15** (training).",
        "body": "generate text, export weights, optional C sample.",
    },
    "17.Phase5CBackward.ipynb": {
        "title": "17 — C backward (MLA + MoE)",
        "before": "notebooks **15–16**.",
        "body": "mirror `train_gpt2.c` backward in `c/deepseek_v2/` — terminal commands.",
    },
    "18.DeepSeekV4Path.ipynb": {
        "title": "18 — DeepSeek-V4 path (hash-MoE + C port)",
        "before": "notebooks **11–17** (V2 + C backward).",
        "body": "V4 building blocks — hash-MoE, SwiGLU, sliding attention map.",
        "dojo": "dojo-grade --lesson C2-L18",
    },
}


def main() -> None:
    for i in range(1, 11):
        paths = list(ROOT.glob(f"{i}.*.ipynb"))
        for p in paths:
            patch_notebook(p)
    patch_nb8_train_skip()
    patch_nb10_gpt2()
    for name, meta in META.items():
        patch_notebook(ROOT / name, meta)
    patch_nb15_train()
    print("done")


if __name__ == "__main__":
    main()
