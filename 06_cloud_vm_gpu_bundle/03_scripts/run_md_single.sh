#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash 03_scripts/run_md_single.sh <structure_id> [auto|start|original]" >&2
  exit 1
fi

STRUCTURE_ID="$1"
INPUT_MODE="${2:-auto}"

BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ENV_DIR="${ENV_DIR:-$BUNDLE_ROOT/.venv}"
TASK_NAME="${UMA_TASK_NAME:-omat}"
DEVICE="${UMA_DEVICE:-cuda}"
INFERENCE_MODE="${UMA_INFERENCE_MODE:-default}"
MD_ENSEMBLE="${MD_ENSEMBLE:-NVT}"
MD_TEMP="${MD_TEMP:-600}"
MD_TIMESTEP="${MD_TIMESTEP:-1.0}"
MD_STEPS="${MD_STEPS:-1000}"
MD_FRICTION="${MD_FRICTION:-0.001}"
MD_SAVE_INTERVAL="${MD_SAVE_INTERVAL:-20}"
RUN_NAME="${RUN_NAME:-md_${MD_TEMP}K_${MD_STEPS}steps}"
PRE_RELAX="${MD_PRE_RELAX:-0}"
PRE_RELAX_STEPS="${MD_PRE_RELAX_STEPS:-50}"
PRE_RELAX_FMAX="${MD_PRE_RELAX_FMAX:-0.1}"

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
PREPARED_DIR="$BUNDLE_ROOT/04_runs/prepared/$STRUCTURE_ID"
OUTPUT_DIR="$BUNDLE_ROOT/04_runs/md/$STRUCTURE_ID/$RUN_NAME"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "UMA environment not found: $ENV_DIR" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "UMA model not found: $MODEL_PATH" >&2
  exit 1
fi

case "$INPUT_MODE" in
  auto)
    if [[ -f "$PREPARED_DIR/POSCAR.start" ]]; then
      INPUT_FILE="$PREPARED_DIR/POSCAR.start"
    elif [[ -f "$INPUT_DIR/POSCAR.start" ]]; then
      INPUT_FILE="$INPUT_DIR/POSCAR.start"
    else
      INPUT_FILE="$INPUT_DIR/POSCAR.original"
    fi
    ;;
  start)
    INPUT_FILE="$INPUT_DIR/POSCAR.start"
    ;;
  original)
    INPUT_FILE="$INPUT_DIR/POSCAR.original"
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

mkdir -p "$OUTPUT_DIR"
PYTHON_TARGET="$ENV_DIR/bin/python"
if [[ ! -x "$PYTHON_TARGET" ]]; then
  echo "Managed Python not found: $PYTHON_TARGET" >&2
  exit 1
fi

echo "[INFO] structure_id    = $STRUCTURE_ID"
echo "[INFO] input_file      = $INPUT_FILE"
echo "[INFO] output_dir      = $OUTPUT_DIR"
echo "[INFO] device          = $DEVICE"
echo "[INFO] inference_mode  = $INFERENCE_MODE"
echo "[INFO] ensemble        = $MD_ENSEMBLE"
echo "[INFO] temp_K          = $MD_TEMP"
echo "[INFO] timestep_fs     = $MD_TIMESTEP"
echo "[INFO] steps           = $MD_STEPS"
echo "[INFO] save_interval   = $MD_SAVE_INTERVAL"
echo "[INFO] pre_relax       = $PRE_RELAX"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"$PYTHON_TARGET" -u "$BUNDLE_ROOT/03_scripts/run_uma_md_direct.py" \
  --structure "$INPUT_FILE" \
  --model "$MODEL_PATH" \
  --output "$OUTPUT_DIR" \
  --task "$TASK_NAME" \
  --device "$DEVICE" \
  --inference-mode "$INFERENCE_MODE" \
  --ensemble "$MD_ENSEMBLE" \
  --temp "$MD_TEMP" \
  --timestep "$MD_TIMESTEP" \
  --steps "$MD_STEPS" \
  --friction "$MD_FRICTION" \
  --save-interval "$MD_SAVE_INTERVAL" \
  --name "$RUN_NAME" \
  $( [[ "$PRE_RELAX" == "1" ]] && printf '%s' "--pre-relax" ) \
  --pre-relax-steps "$PRE_RELAX_STEPS" \
  --pre-relax-fmax "$PRE_RELAX_FMAX"
