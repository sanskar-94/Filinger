#!/usr/bin/env bash
# Phase 3 on a Mac from a fresh clone, in one command: a Python environment, the synthetic lake, the 3B MLX fine tune
# with its evaluation, gate and registration (scripts/mac_finetune.sh), then the results copied to eval/mac_3b/ so they
# can be committed, where the docs and the BI export (flab bi-export --phase3-3b) read them.
#
#   bash scripts/mac_full_run.sh          # about 3 to 4 hours on an 8 GB M1 or M2: plugged in, other apps closed
#   MODEL=mlx-community/Qwen2.5-1.5B-Instruct-4bit NAME=qwen1p5b-4bit bash scripts/mac_full_run.sh   # if 3B runs out of memory
#   SMOKE=1 VENV=path/to/venv DEST=/tmp/out bash scripts/mac_full_run.sh                            # the same steps on the small model
#
# Then publish the results on the branch:  git add eval/mac_3b && git commit -m "3B MLX fine tune results" && git push
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${SMOKE:-0}" != "1" ] && { [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; }; then
  echo "this script is for an Apple Silicon Mac; on a Linux GPU machine use scripts/gpu_finetune.sh"
  exit 1
fi
if pgrep -f "mlx_lm lora|flab llm-eval" >/dev/null 2>&1; then  # the two steps that hold a model in memory
  echo "a fine tune is already running on this machine; let it finish (two at once do not fit in 8 GB)"
  exit 1
fi

# 1. Python 3.10 or newer (macOS ships 3.9), in its own virtual environment
VENV=${VENV:-.venv-flab}
if [ ! -x "$VENV/bin/python" ]; then
  PY=""
  for c in python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
      PY=$(command -v "$c")
      break
    fi
  done
  if [ -z "$PY" ]; then
    echo "Python 3.10 or newer is needed: install it from https://www.python.org/downloads/macos/ or with brew install python@3.12"
    exit 1
  fi
  echo "creating $VENV with $PY ($("$PY" --version 2>&1))"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e ".[ml,serve,mlx]"
python -c "import mlx.core as mx, mlx_lm; print('mlx', mx.__version__, 'device', mx.default_device())"

# 2. the synthetic world and its lake (the fine tune data and the gold filings come from the lake)
if [ ! -e data/lake/bronze_index.parquet ]; then
  flab synth --out data/synthetic --companies 60
  flab local --index data/synthetic/index.parquet --prices data/synthetic/prices.parquet --lake data/lake
fi

# 3. data, LoRA fine tune, evaluation against production and the base model, gate, registration
MODEL=${MODEL:-mlx-community/Qwen2.5-3B-Instruct-4bit}
NAME=${NAME:-qwen3b-4bit}
OUT=${OUT:-artifacts/mac_run}
if [ "${SMOKE:-0}" = "1" ]; then MODEL=artifacts/tiny-base; OUT=${OUT_SMOKE:-artifacts/mac_run_smoke}; fi
if [ "$NAME" = "qwen3b-4bit" ]; then DEST=${DEST:-eval/mac_3b}; else DEST=${DEST:-eval/mac_$NAME}; fi
export MODEL NAME OUT
KEEP_AWAKE=()
if command -v caffeinate >/dev/null 2>&1; then KEEP_AWAKE=(caffeinate -i); fi  # no sleep while it runs
mkdir -p artifacts
START=$(date +%s)
${KEEP_AWAKE[@]+"${KEEP_AWAKE[@]}"} bash scripts/mac_finetune.sh 2>&1 | tee artifacts/mac_run.log
WALL=$(( $(date +%s) - START ))

# 4. the results, small enough to commit: report, metrics, gate verdict, registration, end of the training log
mkdir -p "$DEST"
cp "$OUT/eval/report.md" "$OUT/eval/eval.json" "$OUT/gate.txt" "$OUT/register.json" "$DEST/"
tail -n 40 "$OUT/train.log" > "$DEST/train_tail.txt"
{
  echo "wall_seconds: $WALL"
  echo "model: $MODEL"
  echo "machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m), $(sysctl -n hw.memsize 2>/dev/null || echo unknown) bytes"
  echo "os: $(sw_vers -productVersion 2>/dev/null || uname -sr)"
  echo "python: $(python --version 2>&1)"
  echo "mlx: $(python -c 'import mlx.core as mx; print(mx.__version__)'), mlx-lm: $(python -m pip show mlx-lm 2>/dev/null | awk '/^Version/ {print $2}')"
  echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
} > "$DEST/machine.txt"

echo
echo "finished in $((WALL / 60)) minutes; results in $DEST:"
ls "$DEST"
echo "publish them with: git add $DEST && git commit -m \"3B MLX fine tune results\" && git push"
