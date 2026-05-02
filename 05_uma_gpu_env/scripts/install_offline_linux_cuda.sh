#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${1:-$HOME/uma-gpu-env}"
WHEELHOUSE="$ROOT_DIR/wheelhouse_cuda"

echo "[INFO] This is a placeholder installer for a future CUDA-capable UMA env."
echo "[INFO] Bundle root: $ROOT_DIR"
echo "[INFO] Target env:  $ENV_PREFIX"
echo "[INFO] Wheelhouse:  $WHEELHOUSE"
echo
echo "[ERROR] GPU bundle has not been assembled yet."
echo "[ERROR] Current blocker: GPU jobs on node3 are failing with launch_failed_requeued_held."
echo "[ERROR] Resolve scheduler/node availability first, then populate wheelhouse_cuda."
exit 1
