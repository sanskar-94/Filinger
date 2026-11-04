#!/usr/bin/env bash
# Phase 3 on an Apple Silicon Mac, including one with 8 GB of memory: LoRA on Qwen2.5 3B Instruct quantized to
# 4 bit, trained with Apple's MLX on the Mac's GPU, then scored with the same harness and judged by the same gate
# as every other system, and registered with its verdict. Half precision 3B weights alone need about 6 GB, which
# an 8 GB Mac cannot spare for training; the 4 bit model is about 2 GB and trains in about 3 GB with these settings.
#
#   pip install -e ".[ml,serve,mlx]"                                         # once, on the Mac
#   bash scripts/mac_finetune.sh                                             # Qwen2.5 3B Instruct, 4 bit
#   MODEL=mlx-community/Qwen2.5-1.5B-Instruct-4bit NAME=qwen1p5b-4bit bash scripts/mac_finetune.sh   # if memory runs short
#   SMOKE=1 bash scripts/mac_finetune.sh                                     # the same steps on the small local model
#
# Expect training to take under an hour and the evaluation (two models on 400 filings) one to two hours on an
# M1 or M2 with 8 GB; close other applications first. MLFLOW_TRACKING_URI sends the registration to MLflow.
set -euo pipefail

MODEL=${MODEL:-mlx-community/Qwen2.5-3B-Instruct-4bit}
NAME=${NAME:-qwen3b-4bit}
DATA=${DATA:-data/llm_dataset_compact}
OUT=${OUT:-artifacts/mac_run}
ITERS=${ITERS:-2000}
LAYERS=${LAYERS:-16}
LIMIT=${LIMIT:-400}
VAL_BATCHES=${VAL_BATCHES:-25}

if [ "${SMOKE:-0}" = "1" ]; then  # every step for real, on the small model pretrained by flab llm-pretrain
  MODEL=artifacts/tiny-base
  NAME=tiny-mlx
  OUT=${OUT_SMOKE:-artifacts/mac_run_smoke}
  ITERS=8
  LAYERS=4
  LIMIT=12
  VAL_BATCHES=2
fi
mkdir -p "$OUT"
echo "model $MODEL, data $DATA, out $OUT, iterations $ITERS"

# 1. data: compact prompts (1,500 characters of evidence) keep each example near 600 tokens, which is what makes
#    8 GB enough; evidence for every positive label is guaranteed to be in the prompt
if [ ! -f "$DATA/train.jsonl" ]; then
  flab llm-dataset --prompt-style compact --budget 1500 --out "$DATA"
fi
flab llm-mlx-data --data "$DATA" --out "$OUT/mlx_data"

# 2. LoRA on the quantized model: loss on the answer only, batch of one, gradient checkpointing for memory
python -m mlx_lm lora --model "$MODEL" --train --data "$OUT/mlx_data" --adapter-path "$OUT/adapter" \
  --fine-tune-type lora --num-layers "$LAYERS" --batch-size 1 --iters "$ITERS" --learning-rate 1e-4 \
  --max-seq-length 1024 --mask-prompt --grad-checkpoint --steps-per-report 25 --steps-per-eval 250 \
  --val-batches "$VAL_BATCHES" --save-every 500 --seed 0 2>&1 | tee "$OUT/train.log"

# 3. evaluate production (rules), the base model and the fine tuned model on the same gold filings
flab llm-eval --gold "$DATA/gold.jsonl" --limit "$LIMIT" --prompt-style compact \
  --systems "rules,mlx:${MODEL}@${NAME}-base,mlx:${MODEL}|${OUT}/adapter@${NAME}-lora" \
  --reference rules --out "$OUT/eval"

# 4. gate, then register the adapter with its verdict; serving it stays a reviewed pull request
if flab gate --production models/production.json --extractor-eval "$OUT/eval/eval.json" --system "${NAME}-lora" \
     | tee "$OUT/gate.txt"; then
  VERDICT=passed
  ALIAS=(--alias challenger)
else
  VERDICT=failed
  ALIAS=()
fi
# ${ALIAS[@]+...}: macOS ships bash 3.2, where an empty array under set -u is an unbound variable error
flab llm-register --adapter "$OUT/adapter" --name filings-extractor --eval "$OUT/eval/eval.json" \
  --system "${NAME}-lora" --prompt-style compact --tag "gate=$VERDICT" --tag framework=mlx --tag "base_model=$MODEL" \
  ${ALIAS[@]+"${ALIAS[@]}"} | tee "$OUT/register.json"

echo "gate $VERDICT for ${NAME}-lora; report $OUT/eval/report.md"
if [ "${STRICT:-0}" = "1" ] && [ "$VERDICT" = "failed" ]; then exit 1; fi
