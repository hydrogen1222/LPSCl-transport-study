#!/bin/bash
#SBATCH --job-name=cp2k_sp
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=72:00:00

set -euo pipefail

INPUT_FILE="1.inp"
OUTPUT_FILE="1.out"
JOB_BASENAME="$(basename "${INPUT_FILE%.*}")"

if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "Input file not found: ${INPUT_FILE}"
  exit 1
fi

ulimit -d unlimited
ulimit -s unlimited
ulimit -t unlimited
ulimit -v unlimited

export SLURM_EXPORT_ENV=ALL

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_RUN_PATH="${LD_RUN_PATH:-}"
export LIBRARY_PATH="${LIBRARY_PATH:-}"
export CPATH="${CPATH:-}"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
export CP_DFLAGS="${CP_DFLAGS:-}"
export CP_CFLAGS="${CP_CFLAGS:-}"
export CP_LDFLAGS="${CP_LDFLAGS:-}"
export CP_LIBS="${CP_LIBS:-}"

set +u
source /home/ctan/cp2k/cp2k-2026.1/tools/toolchain/install/setup
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export ELPA_DEFAULT_OMP_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

EXE="/home/ctan/cp2k/cp2k-2026.1/build/bin/cp2k.psmp"
MPI_TYPE="${CP2K_MPI_TYPE:-pmix}"
LOG_FILE="${JOB_BASENAME}.${SLURM_JOB_ID}.log"

cd "${SLURM_SUBMIT_DIR}"

echo "Starting CP2K single point on $(hostname) at $(date)" | tee "${LOG_FILE}"
echo "Input: ${INPUT_FILE}" | tee -a "${LOG_FILE}"
echo "Output: ${OUTPUT_FILE}" | tee -a "${LOG_FILE}"
echo "MPI ranks: ${SLURM_NTASKS}" | tee -a "${LOG_FILE}"
echo "OMP threads per rank: ${OMP_NUM_THREADS}" | tee -a "${LOG_FILE}"
echo "Launcher: srun --mpi=${MPI_TYPE}" | tee -a "${LOG_FILE}"

srun --mpi="${MPI_TYPE}" --cpu-bind=cores "${EXE}" -i "${INPUT_FILE}" -o "${OUTPUT_FILE}"

echo "Finished CP2K single point on $(hostname) at $(date)" | tee -a "${LOG_FILE}"
