"""Phase 3 commands: llm-dataset, llm-label, llm-train, llm-eval, llm-features."""
from __future__ import annotations

import json
from pathlib import Path


def _inputs(lake: str, budget: int):
    import pandas as pd

    from .dataset import build_inputs

    lake = lake.rstrip("/")
    paragraphs = pd.read_parquet(f"{lake}/silver/paragraphs")
    index = pd.read_parquet(f"{lake}/bronze_index.parquet")
    return build_inputs(paragraphs, index, budget), index


def make_system(spec: str, style: str = "full"):
    """rules | frontier[:model] | vllm:<base_url>|<model>[|guided] | hf:<model_path>[|<adapter_path>] |
    mlx:<model>[|<adapter_path>] | replay:<labels.jsonl>, each optionally followed by @name.

    ``replay`` scores labels produced elsewhere (an annotator, a batch API results file): one JSON object per line
    with ``accession`` and ``signals``."""
    from .providers import AnthropicExtractor, HFExtractor, LabelExtractor, OpenAICompatExtractor, RuleExtractor

    spec, _, label = spec.partition("@")
    kind, _, rest = spec.partition(":")
    if kind == "rules":
        return RuleExtractor()
    if kind == "mlx":  # Apple Silicon (or Linux CPU) through mlx-lm: mlx:<model>[|<adapter>]
        from .mlx_backend import MLXExtractor

        parts = rest.split("|")
        return MLXExtractor(parts[0], parts[1] if len(parts) > 1 else None, style=style, **({"name": label} if label else {}))
    if kind == "replay":
        rows = [json.loads(x) for x in Path(rest).read_text().splitlines() if x.strip()]
        return LabelExtractor({r["accession"]: r["signals"] for r in rows if r.get("signals")}, name=label or "replay")
    if kind == "frontier":
        return AnthropicExtractor(model=rest) if rest else AnthropicExtractor()
    if kind == "vllm":
        parts = rest.split("|")
        return OpenAICompatExtractor(parts[0], parts[1], guided=len(parts) > 2 and parts[2] == "guided", style=style,
                                     **({"name": label} if label else {}))
    if kind == "hf":
        parts = rest.split("|")
        return HFExtractor(parts[0], parts[1] if len(parts) > 1 else None, style=style, max_new_tokens=256,
                           **({"name": label} if label else {}))
    raise ValueError(f"unknown system spec {spec}")


def cmd_dataset(a):
    import pandas as pd

    from .dataset import assign_splits, labels_from_synthetic_truth, write_dataset

    inputs, index = _inputs(a.lake, a.budget)
    if a.labels:  # teacher labels written by llm-label
        rows = [json.loads(x) for x in Path(a.labels).read_text().splitlines() if x.strip()]
        labels = {r["accession"]: r["signals"] for r in rows if r.get("schema_valid")}
    else:
        labels = labels_from_synthetic_truth(pd.read_parquet(a.truth), inputs)
    stats = write_dataset(inputs, labels, assign_splits(index, a.test_years), a.out, a.max_train, style=a.prompt_style)
    print(json.dumps(stats))


def cmd_label(a):
    """Teacher labelling with Claude: realtime (with refusal fallback) or Message Batches (half price)."""
    import random

    from .providers import AnthropicBatchLabeler, AnthropicExtractor

    inputs, _ = _inputs(a.lake, a.budget)
    keys = sorted(inputs)
    random.Random(0).shuffle(keys)
    chosen = [inputs[k] for k in keys[: a.limit]]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "batch":
        lab = AnthropicBatchLabeler(AnthropicExtractor(model=a.model, use_fallbacks=False))
        ids = lab.submit(chosen)
        print("submitted", ids)
        results = []
        for bid in ids:
            lab.wait(bid)
            results += lab.collect(bid, {i.accession: i for i in chosen})
        failed = [inputs[r.accession] for r in results if not r.schema_valid]
        if failed:  # retry declines and errors in realtime, where the server side fallback is available
            ex = AnthropicExtractor(model=a.model)
            retried = {r.accession: r for r in (ex.extract(i) for i in failed)}
            results = [retried.get(r.accession, r) for r in results]
    else:
        ex = AnthropicExtractor(model=a.model)
        results = [ex.extract(i) for i in chosen]
    with open(out, "w") as fh:
        for r in results:
            fh.write(json.dumps(r.to_dict()) + "\n")
    cost = sum(r.cost_usd for r in results)
    ok = sum(r.schema_valid for r in results)
    print(f"{ok}/{len(results)} labelled, ${cost:.2f} total -> {out}")


def cmd_train(a):
    from .train_lora import main

    argv = ["--data", a.data, "--out", a.out] + (["--config", a.config] if a.config else [])
    argv += ["--max-steps", str(a.max_steps)] if a.max_steps else []
    main(argv)


def cmd_eval(a):
    from .dataset import load_gold
    from .eval_harness import evaluate

    gold = load_gold(a.gold)[: a.limit]
    systems = [make_system(s, a.prompt_style) for s in a.systems.split(",")]
    res = evaluate(systems, gold, a.out, reference=a.reference, workers=a.workers, batch_size=a.batch_size)
    print((Path(a.out) / "report.md").read_text())
    return res


def cmd_features(a):
    from .eval_harness import run_system
    from .features import signals_to_features

    inputs, _ = _inputs(a.lake, a.budget)
    gold_like = [(inp, None) for inp in inputs.values()]
    system = make_system(a.system, a.prompt_style)
    if hasattr(system, "extract_many"):  # batched local generation
        res = system.extract_many([inp for inp, _ in gold_like], batch_size=a.batch_size)
    else:
        res = run_system(system, gold_like, workers=a.workers)
    df = signals_to_features(res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"{len(df)} filings, valid {df['llm_valid'].mean():.3f} -> {a.out}")


def _reg_dataset(p):
    p.add_argument("--lake", default="data/lake")
    p.add_argument("--truth", default="data/synthetic/truth.parquet")
    p.add_argument("--labels", help="teacher labels JSONL from llm-label (overrides --truth)")
    p.add_argument("--out", default="data/llm_dataset")
    p.add_argument("--budget", type=int, default=12_000)
    p.add_argument("--test-years", type=int, default=2)
    p.add_argument("--max-train", type=int)
    p.add_argument("--prompt-style", choices=["full", "compact"], default="full")
    p.set_defaults(fn=cmd_dataset)


def _reg_label(p):
    p.add_argument("--lake", default="data/lake")
    p.add_argument("--out", default="data/teacher/labels.jsonl")
    p.add_argument("--model", default=None)
    p.add_argument("--mode", choices=["batch", "realtime"], default="batch")
    p.add_argument("--limit", type=int, default=3000)
    p.add_argument("--budget", type=int, default=12_000)
    p.set_defaults(fn=lambda a: cmd_label(_with_default_model(a)))


def _with_default_model(a):
    from .providers import FRONTIER_MODEL

    a.model = a.model or FRONTIER_MODEL
    return a


def _reg_train(p):
    p.add_argument("--data", default="data/llm_dataset")
    p.add_argument("--out", default="artifacts/lora/run")
    p.add_argument("--config", default="configs/lora_qwen2.5_3b.json")
    p.add_argument("--max-steps", type=int)
    p.set_defaults(fn=cmd_train)


def _reg_eval(p):
    p.add_argument("--gold", default="data/llm_dataset/gold.jsonl")
    p.add_argument("--systems", default="rules", help="comma separated specs, see make_system")
    p.add_argument("--reference", default=None)
    p.add_argument("--out", default="artifacts/phase3")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--prompt-style", choices=["full", "compact"], default="full")
    p.add_argument("--batch-size", type=int, default=1,
                   help="batched generation for local hf: models; latency then becomes amortized batch time")
    p.set_defaults(fn=cmd_eval)


def _reg_features(p):
    p.add_argument("--lake", default="data/lake")
    p.add_argument("--system", default="rules")
    p.add_argument("--out", default="data/lake/gold/llm_signals.parquet")
    p.add_argument("--budget", type=int, default=12_000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--prompt-style", choices=["full", "compact"], default="full")
    p.add_argument("--batch-size", type=int, default=16)
    p.set_defaults(fn=cmd_features)


def cmd_pretrain(a):
    import pandas as pd

    from .dataset import assign_splits
    from .tiny import pretrain

    lake = a.lake.rstrip("/")
    paragraphs = pd.read_parquet(f"{lake}/silver/paragraphs", columns=["accession", "text"])
    index = pd.read_parquet(f"{lake}/bronze_index.parquet")
    splits = assign_splits(index, a.test_years)
    train_acc = set(splits[splits == "train"].index)
    texts = paragraphs[paragraphs["accession"].isin(train_acc)]["text"].tolist()  # never test period text
    extra = []
    if a.dataset and Path(f"{a.dataset}/train.jsonl").exists():  # teach the tokenizer the label format too
        extra = [json.loads(x)["completion"][0]["content"] for x in Path(f"{a.dataset}/train.jsonl").read_text().splitlines()]
    m = pretrain(texts, a.out, vocab_size=a.vocab, seq_len=a.seq_len, epochs=a.epochs, max_steps=a.max_steps,
                 tokenizer_extra=extra)
    print(json.dumps(m, indent=2))


def _reg_pretrain(p):
    p.add_argument("--lake", default="data/lake")
    p.add_argument("--dataset", default="data/llm_dataset_compact")
    p.add_argument("--out", default="artifacts/tiny-base")
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--test-years", type=int, default=2)
    p.set_defaults(fn=cmd_pretrain)


def cmd_gold(a):
    from .review import ReviewStore

    rs = ReviewStore(a.queue, a.reviewed)
    gold = rs.gold()
    Path(a.out).write_text("\n".join(json.dumps(g) for g in gold) + "\n")
    print(json.dumps({"gold": len(gold), **rs.stats()}, indent=2, default=str))


def _reg_gold(p):
    p.add_argument("--queue", default="data/llm_dataset/gold.jsonl")
    p.add_argument("--reviewed", default="data/llm_dataset/reviewed.jsonl")
    p.add_argument("--out", default="data/llm_dataset/gold_reviewed.jsonl")
    p.set_defaults(fn=cmd_gold)


def cmd_register(a):
    """Register a fine tuned adapter as a version of the extraction model, with its gold set scores, and point an
    alias at it. Under MLflow the version is logged into the adapter's training run."""
    from ..serving.registry import MlflowRegistry, get_registry

    adapter = Path(a.adapter)
    train_dir = adapter.parent
    train_metrics = json.loads((train_dir / "train_metrics.json").read_text()) if (train_dir / "train_metrics.json").exists() else {}
    train_cfg = json.loads((train_dir / "train_config.json").read_text()) if (train_dir / "train_config.json").exists() else {}
    renames = {"train_loss": "train_loss", "final_eval_loss": "val_loss", "trainable_pct": "trainable_pct"}
    metrics = {new: train_metrics[old] for old, new in renames.items() if old in train_metrics}
    if a.eval:
        summary = json.loads(Path(a.eval).read_text())["summary"]
        if a.system not in summary:
            raise SystemExit(f"{a.system} not in {a.eval}; systems there: {', '.join(summary)}")
        metrics.update({k: v for k, v in summary[a.system].items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    tags = {"base_model": str(train_cfg.get("base_model", "")), "lora_r": str(train_cfg.get("lora_r", "")),
            "target_modules": str(train_cfg.get("target_modules", "")), "eval_system": a.system or "",
            "prompt_style": a.prompt_style}
    tags.update(dict(t.split("=", 1) for t in (a.tag or [])))
    reg = get_registry()
    run_id = (a.run_id or train_metrics.get("run_id")) if isinstance(reg, MlflowRegistry) else None
    version = reg.register(a.name, adapter, metrics=metrics, tags=tags, kind="adapter", run_id=run_id)
    if a.alias:
        reg.set_alias(a.name, a.alias, version)
    print(json.dumps({"name": a.name, "version": version, "alias": a.alias, "metrics": metrics}, indent=2, default=float))


def _reg_register(p):
    p.add_argument("--adapter", required=True, help="the adapter folder written by llm-train")
    p.add_argument("--name", default="filings-extractor")
    p.add_argument("--alias", default=None, help="champion or challenger")
    p.add_argument("--eval", default=None, help="eval.json from llm-eval")
    p.add_argument("--system", default=None, help="the system name of this adapter in eval.json")
    p.add_argument("--run-id", default=None, help="MLflow training run; read from train_metrics.json by default")
    p.add_argument("--prompt-style", choices=["full", "compact"], default="compact")
    p.add_argument("--tag", action="append", help="key=value recorded on the version, e.g. gate=failed (repeatable)")
    p.set_defaults(fn=cmd_register)


def cmd_mlx_data(a):
    from .mlx_backend import to_mlx_dataset

    print(json.dumps(to_mlx_dataset(a.data, a.out)))


def _reg_mlx_data(p):
    p.add_argument("--data", default="data/llm_dataset_compact", help="a dataset written by llm-dataset")
    p.add_argument("--out", default="data/mlx_dataset")
    p.set_defaults(fn=cmd_mlx_data)


COMMANDS = {
    "llm-pretrain": (_reg_pretrain, "pretrain the small filings language model from scratch (CPU friendly)"),
    "llm-gold": (_reg_gold, "build the gold set from human reviewed labels"),
    "llm-dataset": (_reg_dataset, "build the fine tuning dataset and gold set"),
    "llm-label": (_reg_label, "label filings with the frontier teacher (Claude)"),
    "llm-train": (_reg_train, "LoRA fine tune a small open model"),
    "llm-eval": (_reg_eval, "compare extractors on the gold set"),
    "llm-features": (_reg_features, "run an extractor over the lake and write llm_* features"),
    "llm-register": (_reg_register, "register a fine tuned adapter as a model version with its eval scores"),
    "llm-mlx-data": (_reg_mlx_data, "convert a fine tuning dataset to mlx-lm chat format (Apple Silicon training)"),
}
