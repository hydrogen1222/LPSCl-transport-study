#!/bin/bash
#SBATCH --job-name=uma_md_pilot
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
DEVICE="${UMA_DEVICE:-cpu}"
INFERENCE_MODE="${UMA_INFERENCE_MODE:-default}"
PRE_RELAX="${UMA_MD_PRE_RELAX:-0}"
PRE_RELAX_STEPS="${UMA_MD_PRE_RELAX_STEPS:-50}"
PRE_RELAX_FMAX="${UMA_MD_PRE_RELAX_FMAX:-0.1}"

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
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"

python -u "${SLURM_SUBMIT_DIR}/../../job_scripts/run_uma_md_direct.py"

RESULT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
if [[ -d "${RESULT_DIR}" ]]; then
  cp -Rf "${RESULT_DIR}/." .
fi

touch MD_DONE
