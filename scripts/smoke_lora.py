"""End to end LoRA smoke test that needs no GPU and no model download.

It builds a byte level BPE tokenizer and a two layer Qwen2 style model with random weights from the
synthetic corpus, writes the Phase 3 dataset, runs the real ``train_lora.train`` for a few steps, reloads
the adapter through ``HFExtractor`` and scores it with the eval harness. The numbers are meaningless; the
point is that every interface between dataset, chat template, TRL, PEFT, saving, merging, inference and
evaluation is exercised exactly as on a GPU with Qwen2.5 3B.

    python scripts/smoke_lora.py --work /tmp/lora_smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def build_tiny_model(corpus: list[str], out: Path) -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=3000, special_tokens=specials,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  bos_token=None, unk_token=None)
    tok.chat_template = CHATML
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(cfg)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="artifacts/lora_smoke")
    ap.add_argument("--steps", type=int, default=6)
    a = ap.parse_args(argv)
    work = Path(a.work)

    import pandas as pd

    from filings_lab.llm.dataset import (assign_splits, build_inputs, labels_from_synthetic_truth, load_gold,
                                         write_dataset)
    from filings_lab.llm.eval_harness import evaluate
    from filings_lab.llm.providers import HFExtractor, RuleExtractor
    from filings_lab.llm.train_lora import LoraTrainConfig, train
    from filings_lab.pipeline import local
    from filings_lab.synthetic import SyntheticConfig, generate

    world = generate(SyntheticConfig(n_companies=6, start="2016-01-01", end="2020-12-31", seed=3), work / "synthetic")
    local.run(world["index"], world["prices"], str(work / "lake"), workers=1)
    paragraphs = pd.read_parquet(work / "lake" / "silver" / "paragraphs")
    inputs = build_inputs(paragraphs, world["index"], budget_chars=3000)
    labels = labels_from_synthetic_truth(world["truth"], inputs)
    stats = write_dataset(inputs, labels, assign_splits(world["index"], test_years=1), work / "dataset", max_train=24)
    print("dataset", stats)

    corpus = [json.loads(x)["prompt"][1]["content"] for x in (work / "dataset" / "train.jsonl").read_text().splitlines()]
    corpus += [json.dumps(labels[k]) for k in list(labels)[:200]]
    base = build_tiny_model(corpus, work / "tiny-base")
    cfg = LoraTrainConfig(base_model=str(base), max_steps=a.steps, per_device_train_batch_size=1,
                          gradient_accumulation_steps=1, max_length=2048, eval_steps=3, logging_steps=1,
                          learning_rate=1e-3, gradient_checkpointing=False)
    metrics = train(cfg, str(work / "dataset"), str(work / "run"))
    print("train", json.dumps(metrics, default=float))

    gold = load_gold(work / "dataset" / "gold.jsonl")[:4]
    systems = [RuleExtractor(), HFExtractor(str(base), name="tiny-base", max_new_tokens=48),
               HFExtractor(str(base), str(work / "run" / "adapter"), name="tiny-lora", max_new_tokens=48)]
    res = evaluate(systems, gold, work / "eval", reference="tiny-base")
    for k, v in res["summary"].items():
        print(f"{k:12s} schema_valid {v['schema_valid_rate']:.2f} score {v['extraction_score']:.3f}")
    assert (work / "run" / "adapter" / "adapter_config.json").exists()
    assert (work / "run" / "merged" / "config.json").exists()
    print("smoke test passed")


if __name__ == "__main__":
    main()
