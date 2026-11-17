#!/usr/bin/env bash
# The distillation experiment of phase 3 (docs/phases/phase3_llm.md, "Distillation from a free teacher"). The same
# 3B student is trained on a free hosted teacher's labels and, as the matched control, on the synthetic generator's
# labels for the same 1,000 filings and the same prompts. Each arm gets several training seeds, and every student is
# scored on the same 400 gold filings against the generator's truth, which the teacher never saw. The report sets
# the arms side by side with the seed noise counted, and shows where the two sets of labels disagree.
#
#   flab llm-label --system api:nvidia/nemotron-3-ultra-550b-a55b --lake data/lake --budget 1500 --workers 8 \
#     --only data/llm_dataset_compact/train.jsonl,data/llm_dataset_compact/val.jsonl --limit 1000 \
#     --out data/teacher/nemotron-3-ultra.jsonl            # the teacher's labels: a free key, about six hours
#   bash scripts/distill.sh                                # both arms, four seeds each: about an hour a run, 8 GB M1
#   ARMS=teacher SEEDS="0 1" bash scripts/distill.sh       # part of it; the report takes the runs that exist
set -euo pipefail

LABELS=${LABELS:-data/teacher/nemotron-3-ultra.jsonl}
N=${N:-1000}
ARMS=${ARMS:-generator teacher}
SEEDS=${SEEDS:-0 1 2 3}
MODEL=${MODEL:-mlx-community/Qwen2.5-3B-Instruct-4bit}
SOURCE=${SOURCE:-data/llm_dataset_compact}
LIMIT=${LIMIT:-400}

name() { if [ "$2" = 0 ]; then echo "$1"; else echo "$1_seed$2"; fi; }   # seed 0 keeps the arm's bare name

if [ ! -f "$SOURCE/train.jsonl" ]; then  # the student's dataset, as scripts/mac_finetune.sh builds it
  flab llm-dataset --prompt-style compact --budget 1500 --out "$SOURCE"
fi
for arm in $ARMS; do
  python scripts/distill_datasets.py --which "$arm" --labels "$LABELS" --n "$N" --source "$SOURCE" --out data/distill
  flab llm-mlx-data --data "data/distill/$arm" --out "artifacts/distill/$arm/mlx_data"
  for seed in $SEEDS; do
    run=$(name "$arm" "$seed")
    system=student-$arm-$N
    if [ "$seed" != 0 ]; then system=$system-seed$seed; fi
    echo "== $run: train $(date +%H:%M)"
    mkdir -p "eval/distill/$run"
    # the settings of scripts/mac_finetune.sh, with 1,000 steps (a little over one pass over 861 training
    # filings) and the seed
    python -m mlx_lm lora --model "$MODEL" --train --data "artifacts/distill/$arm/mlx_data" \
      --adapter-path "artifacts/distill/$run/adapter" --fine-tune-type lora --num-layers 16 --batch-size 1 \
      --iters 1000 --learning-rate 1e-4 --max-seq-length 1024 --mask-prompt --grad-checkpoint \
      --steps-per-report 25 --steps-per-eval 250 --val-batches 25 --save-every 500 --seed "$seed" 2>&1 \
      | sed -u 's/\x1b\[[0-9;]*m//g' | tee "eval/distill/$run/train.log"
    echo "== $run: score $(date +%H:%M)"
    flab llm-eval --gold "$SOURCE/gold.jsonl" --limit "$LIMIT" --prompt-style compact \
      --systems "rules,mlx:$MODEL|artifacts/distill/$run/adapter@$system" --reference rules --out "eval/distill/$run"
  done
done

runs() {  # an arm's finished runs, comma separated
  local out="" seed folder
  for seed in $SEEDS; do
    folder=eval/distill/$(name "$1" "$seed")
    if [ -f "$folder/items.parquet" ]; then out="$out${out:+,}$folder"; fi
  done
  echo "$out"
}
python scripts/distill_report.py --arm "teacher=$(runs teacher)" --arm "generator=$(runs generator)" \
  --data data/distill --out eval/distill/report.md
