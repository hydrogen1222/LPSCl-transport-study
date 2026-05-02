#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ENV_DIR="${ENV_DIR:-$BUNDLE_ROOT/.venv}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
UV_PYTHON_VERSION="${UV_PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0}"
FAIRCHEM_INSTALL_MODE="${FAIRCHEM_INSTALL_MODE:-git-main}"
FAIRCHEM_GIT_REF="${FAIRCHEM_GIT_REF:-main}"
FAIRCHEM_GIT_URL="${FAIRCHEM_GIT_URL:-https://github.com/facebookresearch/fairchem.git}"

echo "[INFO] Bundle root:       $BUNDLE_ROOT"
echo "[INFO] Env dir:           $ENV_DIR"
echo "[INFO] uv binary:         $UV_BIN"
echo "[INFO] Python version:    $UV_PYTHON_VERSION"
echo "[INFO] Torch index URL:   $TORCH_INDEX_URL"
echo "[INFO] Torch packages:    $TORCH_PACKAGES"
echo "[INFO] fairchem mode:     $FAIRCHEM_INSTALL_MODE"
echo "[INFO] fairchem git ref:  $FAIRCHEM_GIT_REF"
echo "[INFO] fairchem git url:  $FAIRCHEM_GIT_URL"

if [[ ! -x "$UV_BIN" ]]; then
  echo "[INFO] uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv installation failed." >&2
  exit 1
fi

uv --version
uv python install "$UV_PYTHON_VERSION"
uv venv --python "$UV_PYTHON_VERSION" "$ENV_DIR"

PYTHON_TARGET="$ENV_DIR/bin/python"
if [[ ! -x "$PYTHON_TARGET" ]]; then
  echo "[ERROR] uv venv creation failed: $PYTHON_TARGET missing" >&2
  exit 1
fi

uv pip install --python "$PYTHON_TARGET" --index-url "$TORCH_INDEX_URL" $TORCH_PACKAGES

if [[ "$FAIRCHEM_INSTALL_MODE" == "pypi" ]]; then
  uv pip install --python "$PYTHON_TARGET" fairchem-core==2.14.0
else
  uv pip install --python "$PYTHON_TARGET" "git+${FAIRCHEM_GIT_URL}@${FAIRCHEM_GIT_REF}#subdirectory=packages/fairchem-core"
fi

uv pip install --python "$PYTHON_TARGET" -e "$BUNDLE_ROOT/02_runtime"

"$PYTHON_TARGET" - <<'PY'
import torch
import fairchem.core
import umakit

print("[INFO] torch =", torch.__version__)
print("[INFO] torch.cuda.is_available() =", torch.cuda.is_available())
print("[INFO] torch.version.cuda =", torch.version.cuda)
print("[INFO] fairchem.core import OK")
print("[INFO] umakit import OK")
if torch.cuda.is_available():
    print("[INFO] GPU 0 =", torch.cuda.get_device_name(0))
PY

echo "[INFO] uv-managed GPU environment setup completed."
