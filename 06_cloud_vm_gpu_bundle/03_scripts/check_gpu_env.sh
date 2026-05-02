#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ENV_DIR="${ENV_DIR:-$BUNDLE_ROOT/.venv}"
export PATH="$HOME/.local/bin:$PATH"

echo "[INFO] Bundle root: $BUNDLE_ROOT"
echo "[INFO] Env dir:     $ENV_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found." >&2
  exit 1
fi

nvidia-smi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "[ERROR] Python environment not found: $ENV_DIR" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv not found in PATH." >&2
  exit 1
fi

echo "[INFO] uv = $(uv --version)"

"$ENV_DIR/bin/python" - <<'PY'
import sys
import torch
import fairchem.core
import umakit

print("[INFO] python =", sys.version.replace("\n", " "))
print("[INFO] torch =", torch.__version__)
print("[INFO] torch.version.cuda =", torch.version.cuda)
print("[INFO] torch.cuda.is_available() =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[INFO] cuda.device_count =", torch.cuda.device_count())
    print("[INFO] GPU 0 name =", torch.cuda.get_device_name(0))
print("[INFO] fairchem.core import OK")
print("[INFO] umakit import OK")
PY
