#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash 03_scripts/run_opt_single.sh <structure_id> [original|start]" >&2
  exit 1
fi

STRUCTURE_ID="$1"
INPUT_MODE="${2:-start}"

BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ENV_DIR="${ENV_DIR:-$BUNDLE_ROOT/.venv}"
TASK_NAME="${UMA_TASK_NAME:-omat}"
DEVICE="${UMA_DEVICE:-cuda}"
OPT_FMAX="${OPT_FMAX:-0.10}"
OPT_STEPS="${OPT_STEPS:-300}"
OPTIMIZER="${OPTIMIZER:-FIRE}"
RUN_NAME="${RUN_NAME:-opt_fmax${OPT_FMAX}_steps${OPT_STEPS}}"

if [[ -n "${UMA_MODEL_PATH:-}" ]]; then
  MODEL_PATH="$UMA_MODEL_PATH"
elif [[ -f "$HOME/models/uma-s-1p2.pt" ]]; then
  MODEL_PATH="$HOME/models/uma-s-1p2.pt"
elif [[ -f "$HOME/models/uma/uma-s-1p2.pt" ]]; then
  MODEL_PATH="$HOME/models/uma/uma-s-1p2.pt"
else
  MODEL_PATH="$HOME/models/uma-s-1p2.pt"
fi

INPUT_DIR="$BUNDLE_ROOT/01_inputs/$STRUCTURE_ID"
OUTPUT_PARENT="$BUNDLE_ROOT/04_runs/opt/$STRUCTURE_ID"
OUTPUT_DIR="$OUTPUT_PARENT/$RUN_NAME"
PREPARED_DIR="$BUNDLE_ROOT/04_runs/prepared/$STRUCTURE_ID"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "UMA environment not found: $ENV_DIR" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "UMA model not found: $MODEL_PATH" >&2
  exit 1
fi

case "$INPUT_MODE" in
  original)
    INPUT_FILE="$INPUT_DIR/POSCAR.original"
    ;;
  start|auto)
    if [[ -f "$PREPARED_DIR/POSCAR.start" ]]; then
      INPUT_FILE="$PREPARED_DIR/POSCAR.start"
    else
      INPUT_FILE="$INPUT_DIR/POSCAR.start"
    fi
    ;;
  *)
    echo "Unknown input mode: $INPUT_MODE" >&2
    exit 1
    ;;
esac

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Input structure not found: $INPUT_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_PARENT" "$PREPARED_DIR"
PYTHON_TARGET="$ENV_DIR/bin/python"
if [[ ! -x "$PYTHON_TARGET" ]]; then
  echo "Managed Python not found: $PYTHON_TARGET" >&2
  exit 1
fi

echo "[INFO] structure_id = $STRUCTURE_ID"
echo "[INFO] input_file   = $INPUT_FILE"
echo "[INFO] output_dir   = $OUTPUT_DIR"
echo "[INFO] device       = $DEVICE"
echo "[INFO] fmax         = $OPT_FMAX"
echo "[INFO] max_steps    = $OPT_STEPS"
echo "[INFO] optimizer    = $OPTIMIZER"

"$PYTHON_TARGET" -u -m umakit.cli opt "$INPUT_FILE" \
  --model "$MODEL_PATH" \
  --task "$TASK_NAME" \
  --device "$DEVICE" \
  --fmax "$OPT_FMAX" \
  --max-steps "$OPT_STEPS" \
  --optimizer "$OPTIMIZER" \
  --output "$OUTPUT_PARENT" \
  --name "$RUN_NAME"

if [[ -f "$OUTPUT_DIR/CONTCAR" ]]; then
  cp -f "$OUTPUT_DIR/CONTCAR" "$PREPARED_DIR/POSCAR.start"
  echo "[INFO] Prepared structure updated: $PREPARED_DIR/POSCAR.start"
else
  echo "[ERROR] Optimization finished without CONTCAR: $OUTPUT_DIR/CONTCAR" >&2
  exit 1
fi
