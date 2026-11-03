"""LoRA / QLoRA supervised fine tuning of a small open model for structured signal extraction.

Recipe (docs/RESEARCH.md section 5): rank 16, alpha 32, dropout 0.05 on every linear projection; learning
rate 2e-4 with cosine decay; two epochs; loss on the assistant completion only (the prompt is long and
identical in structure, so training on it would waste capacity); 4 bit NF4 base weights when ``qlora`` is
set and a GPU is present.

Usage::

    python -m filings_lab.llm.train_lora --config configs/lora_qwen2.5_3b.json \
        --data data/llm_dataset --out artifacts/lora/qwen2.5-3b-filings
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LoraTrainConfig:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    learning_rate: float = 2e-4
    num_train_epochs: float = 2.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_length: int = 6144
    gradient_checkpointing: bool = True
    eval_steps: int = 100
    logging_steps: int = 10
    seed: int = 0
    merge_after_training: bool = True
    report_to: list[str] = field(default_factory=lambda: ["none"])

    @classmethod
    def load(cls, path: str | None, **overrides) -> LoraTrainConfig:
        data = json.loads(Path(path).read_text()) if path else {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


def _reporters(report_to: list[str]) -> list[str]:
    """MLflow only when a tracking server is configured, as everywhere else in the lab: without one, the
    transformers callback falls back to a local store that may not support the registry and stops the run."""
    from .. import tracking

    if "mlflow" in report_to and not tracking.mlflow_enabled():
        report_to = [r for r in report_to if r != "mlflow"] or ["none"]
    return report_to


def _dtype_kwarg() -> str:
    """transformers 5 renamed ``torch_dtype`` to ``dtype``."""
    import inspect

    from transformers import modeling_utils

    sig = inspect.signature(modeling_utils.PreTrainedModel.from_pretrained)
    return "dtype" if "dtype" in sig.parameters or "torch_dtype" not in sig.parameters else "torch_dtype"


def _compat(config_cls, kwargs: dict) -> dict:
    """Map arguments across TRL and transformers versions and drop ones this version does not know."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(config_cls)}
    if "warmup_ratio" in kwargs and "warmup_ratio" not in fields:
        kwargs["warmup_steps"] = kwargs.pop("warmup_ratio")  # newer versions take a float ratio here
    unknown = [k for k in kwargs if k not in fields]
    if unknown:
        print(f"ignoring arguments unknown to this TRL version: {unknown}")
    return {k: v for k, v in kwargs.items() if k in fields}


def train(cfg: LoraTrainConfig, data_dir: str, out_dir: str) -> dict:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    files = {"train": f"{data_dir}/train.jsonl"}
    if Path(f"{data_dir}/val.jsonl").exists() and Path(f"{data_dir}/val.jsonl").stat().st_size:
        files["validation"] = f"{data_dir}/val.jsonl"
    ds = load_dataset("json", data_files=files)
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ("prompt", "completion")])

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs: dict = {_dtype_kwarg(): torch.bfloat16 if bf16 else torch.float32}
    if cfg.qlora and cuda:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
    if cfg.qlora and cuda:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)

    peft_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                          target_modules=cfg.target_modules, task_type="CAUSAL_LM", bias="none")
    has_val = "validation" in ds
    args = SFTConfig(**_compat(SFTConfig, dict(
        output_dir=out_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        max_length=cfg.max_length,
        completion_only_loss=True,
        packing=False,
        gradient_checkpointing=cfg.gradient_checkpointing and cuda,
        bf16=bf16,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps" if has_val else "no",
        eval_steps=cfg.eval_steps if has_val else None,
        save_strategy="steps" if has_val else "epoch",
        save_steps=cfg.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss" if has_val else None,
        report_to=_reporters(cfg.report_to),
        seed=cfg.seed,
    )))
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds["train"],
                         eval_dataset=ds["validation"] if has_val else None,
                         processing_class=tok, peft_config=peft_cfg)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    result = trainer.train()
    adapter_dir = Path(out_dir) / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    metrics = {"train_loss": result.training_loss, "steps": result.global_step, "trainable_params": trainable,
               "total_params": total, "trainable_pct": 100 * trainable / total}
    if has_val:
        metrics.update({f"final_{k}": v for k, v in trainer.evaluate().items()})
    if cfg.merge_after_training and not (cfg.qlora and cuda):
        merged = trainer.model.merge_and_unload()  # one artefact vLLM can serve without the LoRA runtime
        merged.save_pretrained(Path(out_dir) / "merged", safe_serialization=True)
        tok.save_pretrained(Path(out_dir) / "merged")
    (Path(out_dir) / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (Path(out_dir) / "train_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model")
    ap.add_argument("--max-steps", type=int)
    a = ap.parse_args(argv)
    cfg = LoraTrainConfig.load(a.config, base_model=a.base_model, max_steps=a.max_steps)
    from .. import tracking

    with tracking.run("phase3-lora", Path(a.out).name) as tr:
        tr.params(asdict(cfg))
        m = train(cfg, a.data, a.out)
        tr.metrics({k: v for k, v in m.items() if isinstance(v, (int, float))})
        tr.artifacts(str(Path(a.out) / "adapter"), "adapter")
        m["run_id"] = tr.id  # flab llm-register logs the model version into this run: training lineage
        (Path(a.out) / "train_metrics.json").write_text(json.dumps(m, indent=2, default=float))
    print(json.dumps(m, indent=2, default=float))


if __name__ == "__main__":
    main()
