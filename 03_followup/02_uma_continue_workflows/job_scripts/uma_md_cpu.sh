#!/bin/bash
#SBATCH --job-name=uma_md
#SBATCH --partition=workq
#SBATCH --nodelist=baifq-hpc141
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00

set -euo pipefail

STRUCTURE_FILE="${UMA_STRUCTURE_FILE:-POSCAR}"
MODEL_PATH="${UMA_MODEL_PATH:-/home/ctan/uma-m-1p1.pt}"
ENV_PREFIX="${UMA_ENV_PREFIX:-/home/ctan/uma-offline-env}"
TASK_NAME="${UMA_TASK_NAME:-omat}"
ENSEMBLE="${UMA_MD_ENSEMBLE:-NVT}"
TEMP_K="${UMA_MD_TEMP:-600}"
TIMESTEP_FS="${UMA_MD_TIMESTEP:-1.0}"
STEPS="${UMA_MD_STEPS:-5000}"
SAVE_INTERVAL="${UMA_MD_SAVE_INTERVAL:-20}"
RUN_NAME="${UMA_RUN_NAME:-uma_md}"
OUTPUT_ROOT="${UMA_OUTPUT_ROOT:-${SLURM_SUBMIT_DIR}}"

if [[ ! -f "${STRUCTURE_FILE}" ]]; then
  echo "Input structure not found: ${STRUCTURE_FILE}" >&2
  exit 1
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "UMA offline environment not found: ${ENV_PREFIX}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "UMA model not found: ${MODEL_PATH}" >&2
  exit 1
fi

source "${ENV_PREFIX}/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export PYTHONNOUSERSITE=1

cd "${SLURM_SUBMIT_DIR}"

uma_calc md "${STRUCTURE_FILE}" \
  --model "${MODEL_PATH}" \
  --task "${TASK_NAME}" \
  --device cpu \
  --ensemble "${ENSEMBLE}" \
  --temp "${TEMP_K}" \
  --timestep "${TIMESTEP_FS}" \
  --steps "${STEPS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --output "${OUTPUT_ROOT}" \
  --name "${RUN_NAME}"

RESULT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
if [[ -d "${RESULT_DIR}" ]]; then
  cp -Rf "${RESULT_DIR}/." .
fi

touch MD_DONE
