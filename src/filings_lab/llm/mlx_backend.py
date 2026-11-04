"""Fine tuning and inference on Apple Silicon with MLX (``mlx-lm``).

An 8 GB Mac cannot hold a 3B model in half precision for training, but it can train LoRA adapters on the same
model quantized to 4 bit, which MLX runs on the Mac's GPU. This module converts the lab's fine tuning data to
mlx-lm's chat format and gives the eval harness an extractor for MLX models, so a Mac run is scored by the same
harness and judged by the same gate as every other system. mlx-lm also runs on Linux CPUs, which is how the smoke
test exercises this path without a Mac.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .prompts import ExtractionInput, chat_messages
from .providers import ExtractionResult, Extractor, _result

SPLITS = {"train": "train", "val": "valid", "test": "test"}  # mlx-lm calls the validation split valid.jsonl


def to_mlx_dataset(src_dir: str | Path, out_dir: str | Path) -> dict[str, int]:
    """Lab SFT records (``prompt`` and ``completion`` message lists) to mlx-lm chat records (``messages``). With
    ``--mask-prompt`` the loss covers only the assistant's answer, as with TRL's completion only loss."""
    src, out = Path(src_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, target in SPLITS.items():
        f = src / f"{name}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        (out / f"{target}.jsonl").write_text(
            "\n".join(json.dumps({"messages": r["prompt"] + r["completion"]}) for r in rows) + "\n")
        counts[target] = len(rows)
    return counts


class MLXExtractor(Extractor):
    """Greedy generation with an MLX model and, optionally, a LoRA adapter trained by ``mlx_lm lora``.

    The model loads on first use and ``release`` drops it, so an evaluation of several systems on an 8 GB Mac
    holds one model in memory at a time."""

    def __init__(self, model_path: str, adapter_path: str | None = None, name: str | None = None,
                 style: str = "compact", max_new_tokens: int = 256):
        self.model_path, self.adapter_path = model_path, adapter_path
        self.name = name or f"mlx:{Path(model_path).name}" + ("+lora" if adapter_path else "")
        self.style, self.max_new_tokens = style, max_new_tokens
        self.model = self.tok = None

    def _load(self) -> None:
        if self.model is None:
            from mlx_lm import load

            self.model, self.tok = load(self.model_path, adapter_path=self.adapter_path)

    def release(self) -> None:
        self.model = self.tok = None
        import gc

        gc.collect()

    def _prompt_ids(self, inp: ExtractionInput) -> list[int]:
        ids = self.tok.apply_chat_template(chat_messages(inp, style=self.style), add_generation_prompt=True, tokenize=True)
        return list(ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids)

    def extract(self, inp: ExtractionInput) -> ExtractionResult:
        from mlx_lm import generate

        self._load()
        ids = self._prompt_ids(inp)  # token ids, so no extra special tokens are added on top of the template
        t0 = time.perf_counter()
        text = generate(self.model, self.tok, ids, max_tokens=self.max_new_tokens, verbose=False)
        return _result(self.name, inp, text, time.perf_counter() - t0, input_tokens=len(ids))
