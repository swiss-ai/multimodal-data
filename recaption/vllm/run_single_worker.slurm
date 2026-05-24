#!/usr/bin/env bash
#SBATCH --job-name=recaption-single-worker
#SBATCH --array=0-31
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
: "${RECAPTION_LOADER:?RECAPTION_LOADER must be set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set}"
: "${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT must be set}"

export RECAPTION_CACHE_ROOT_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp/${USER}/recaption-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}}"

mkdir -p logs

echo "Array job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
echo "Started at $(date --iso-8601=seconds)"
echo "Loader ${RECAPTION_LOADER}"
echo "Cache root base ${RECAPTION_CACHE_ROOT_BASE}"
echo "Using uenv prgenv-gnu/24.11:v1 with modules view"

uenv run prgenv-gnu/24.11:v1 --view=modules -- bash -lc '
module load cuda gcc cmake ninja
nvcc --version
ninja --version
srun \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gpus-per-task=1 \
  --cpus-per-task=72 \
  --cpu-bind=cores \
  --gpu-bind=single:1 \
  --kill-on-bad-exit=1 \
  --label \
  bash -lc '"'"'
cache_root="${RECAPTION_CACHE_ROOT_BASE}/worker-0"
xdg_cache_root="${cache_root}/xdg-cache"
vllm_cache_root="${cache_root}/vllm-cache"
vllm_rpc_base_path="${cache_root}/vllm-rpc"

export FLASHINFER_WORKSPACE_BASE="${cache_root}/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
export TRITON_CACHE_DIR="${cache_root}/triton"
export XDG_CACHE_HOME="${xdg_cache_root}"
export VLLM_CACHE_ROOT="${vllm_cache_root}"
export VLLM_RPC_BASE_PATH="${vllm_rpc_base_path}"
export VLLM_PORT=$((39000 + SLURM_ARRAY_TASK_ID * 4))

mkdir -p \
  "${FLASHINFER_WORKSPACE_BASE}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${VLLM_CACHE_ROOT}" \
  "${VLLM_RPC_BASE_PATH}"

echo "Logical task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Cache root ${cache_root}"
echo "VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
echo "VLLM_RPC_BASE_PATH=${VLLM_RPC_BASE_PATH}"
echo "VLLM_PORT=${VLLM_PORT}"

.venv/bin/python main.py
'"'"'
'

echo "Finished at $(date --iso-8601=seconds)"
