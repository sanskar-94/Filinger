#!/usr/bin/env bash
# Phase 3 as one job: teacher labels (when an API key is present), the dataset, the LoRA or QLoRA fine tune, an
# evaluation of the base, fine tuned, frontier and production systems on the production gold set, the gate, and
# registration of the adapter with its verdict. Every step is a flab command; this script only orders them.
#
#   bash scripts/gpu_finetune.sh                                         # Qwen2.5 3B Instruct, LoRA, one 24 GB GPU
#   CONFIG=configs/qlora_llama3.1_8b.json bash scripts/gpu_finetune.sh   # Llama 3.1 8B Instruct, 4 bit QLoRA
#   SMOKE=1 bash scripts/gpu_finetune.sh                                 # the same steps on the CPU sized model
#
# Environment: ANTHROPIC_API_KEY adds teacher labels and the frontier system; MLFLOW_TRACKING_URI sends runs and
# the registered version to MLflow (otherwise the file registry); STRICT=1 exits non zero when the gate fails.
# The base model is read from the model hub (HF_TOKEN for gated models such as Llama).
set -euo pipefail

CONFIG=${CONFIG:-configs/lora_qwen2.5_3b.json}
LAKE=${LAKE:-data/lake}
DATA=${DATA:-data/llm_dataset}
OUT=${OUT:-artifacts/gpu_run}
LIMIT=${LIMIT:-400}
BATCH=${BATCH:-8}
STYLE=full
TRAIN_ARGS=()

if [ "${SMOKE:-0}" = "1" ]; then  # every step for real, on the small model pretrained by flab llm-pretrain
  CONFIG=configs/lora_tiny_champion.json
  DATA=data/llm_dataset_compact
  OUT=${OUT_SMOKE:-artifacts/gpu_run_smoke}
  LIMIT=24
  BATCH=8
  STYLE=compact
  TRAIN_ARGS=(--max-steps 4)
fi

BASE=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['base_model'])" "$CONFIG")
NAME=${NAME:-$(basename "$BASE" | tr '[:upper:]._' '[:lower:]--')}
CANDIDATE="${NAME}-lora"
mkdir -p "$OUT"
echo "base $BASE, config $CONFIG, data $DATA, out $OUT"

# 1. labels: the frontier teacher when a key is present, otherwise the labels the dataset already carries
if [ -n "${ANTHROPIC_API_KEY:-}" ] && [ "${SMOKE:-0}" != "1" ]; then
  flab llm-label --lake "$LAKE" --mode batch --limit "${TEACHER_LIMIT:-3000}" --out "$OUT/teacher_labels.jsonl"
  flab llm-dataset --lake "$LAKE" --labels "$OUT/teacher_labels.jsonl" --out "$DATA"
elif [ ! -f "$DATA/train.jsonl" ]; then
  flab llm-dataset --lake "$LAKE" --out "$DATA" --prompt-style "$STYLE"
fi

# 2. fine tune
flab llm-train --config "$CONFIG" --data "$DATA" --out "$OUT/$NAME" "${TRAIN_ARGS[@]}"

# 3. evaluate: production (rules), the base model, the fine tuned model, and the frontier model when available,
#    all on the same gold filings so the gate compares like with like
SYSTEMS="rules,hf:${BASE}@${NAME}-base,hf:${BASE}|${OUT}/${NAME}/adapter@${CANDIDATE}"
if [ -n "${ANTHROPIC_API_KEY:-}" ] && [ "${SMOKE:-0}" != "1" ]; then SYSTEMS="${SYSTEMS},frontier@frontier"; fi
flab llm-eval --gold "$DATA/gold.jsonl" --limit "$LIMIT" --prompt-style "$STYLE" --batch-size "$BATCH" \
  --systems "$SYSTEMS" --reference rules --out "$OUT/eval"

# 4. gate, then register the adapter with its verdict; promotion to production stays a reviewed pull request
if flab gate --production models/production.json --extractor-eval "$OUT/eval/eval.json" --system "$CANDIDATE" \
     | tee "$OUT/gate.txt"; then
  VERDICT=passed
  ALIAS=(--alias challenger)
else
  VERDICT=failed
  ALIAS=()
fi
flab llm-register --adapter "$OUT/$NAME/adapter" --name filings-extractor --eval "$OUT/eval/eval.json" \
  --system "$CANDIDATE" --prompt-style "$STYLE" --tag "gate=$VERDICT" "${ALIAS[@]}" | tee "$OUT/register.json"

echo "gate $VERDICT for $CANDIDATE; report $OUT/eval/report.md"
if [ "${STRICT:-0}" = "1" ] && [ "$VERDICT" = "failed" ]; then exit 1; fi
