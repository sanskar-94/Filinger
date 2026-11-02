"""A small Qwen2 style model pretrained from scratch on the filings corpus.

Why it exists: the Phase 3 recipe targets Qwen2.5 3B or Llama 3.2 3B on a GPU. Without a GPU or model
downloads, the same pipeline still has to be executed for real, so this module trains a model small enough for
a CPU (about six million parameters) through the same stages: tokenizer, causal language model pretraining
on filing text, then the LoRA fine tune in ``train_lora`` and the eval harness. Only the base model differs.

Leakage: pretraining text comes only from filings in the training split; test period filings are never seen.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def build_tokenizer(texts, vocab_size: int = 4096):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(texts, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  bos_token=None, unk_token=None)
    tok.chat_template = CHATML
    return tok


def init_model(tok, hidden: int = 256, layers: int = 6, heads: int = 8, kv_heads: int = 4, intermediate: int = 768,
               max_positions: int = 2048, seed: int = 0):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=hidden, intermediate_size=intermediate, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=kv_heads, max_position_embeddings=max_positions,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True,
                      rope_theta=10000.0)
    return Qwen2ForCausalLM(cfg)


def pack(tok, texts, seq_len: int):
    """Concatenate documents separated by the end of text token and cut into fixed length blocks."""
    ids: list[int] = []
    sep = tok.convert_tokens_to_ids("<|endoftext|>")
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        ids.append(sep)
    n = len(ids) // seq_len
    return [ids[i * seq_len:(i + 1) * seq_len] for i in range(n)]


def pretrain(texts: list[str], out_dir: str, vocab_size: int = 4096, seq_len: int = 512, epochs: float = 1.0,
             batch_size: int = 16, learning_rate: float = 3e-3, max_steps: int = -1, seed: int = 0, threads: int = 4,
             tokenizer_extra: list[str] | None = None) -> dict:
    import torch
    from datasets import Dataset
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    torch.set_num_threads(threads)
    out = Path(out_dir)
    tok = build_tokenizer([*texts, *(tokenizer_extra or [])], vocab_size)
    model = init_model(tok, seed=seed)
    blocks = pack(tok, texts, seq_len)
    split = max(1, int(len(blocks) * 0.02))
    train, held = Dataset.from_dict({"input_ids": blocks[split:]}), Dataset.from_dict({"input_ids": blocks[:split]})
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate, num_train_epochs=epochs, max_steps=max_steps, warmup_steps=0.05,
        lr_scheduler_type="cosine", weight_decay=0.01, logging_steps=20, save_strategy="no", eval_strategy="no",
        report_to=["none"], seed=seed, dataloader_num_workers=0, use_cpu=not torch.cuda.is_available())
    trainer = Trainer(model=model, args=args, train_dataset=train,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    result = trainer.train()
    ev = trainer.evaluate(held)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    metrics = {"params": sum(p.numel() for p in model.parameters()), "blocks": len(blocks), "seq_len": seq_len,
               "train_tokens": len(blocks) * seq_len, "steps": result.global_step, "train_loss": result.training_loss,
               "heldout_loss": ev["eval_loss"], "heldout_perplexity": math.exp(ev["eval_loss"])}
    (out / "pretrain_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
