#!/usr/bin/env bash
#SBATCH --job-name=recaption-openimages-v7-dense-full
#SBATCH --array=0-31
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

export WORKERS_PER_NODE=4
export CPUS_PER_WORKER=72
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
: "${RECAPTION_LOADER:?RECAPTION_LOADER must be set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set}"
: "${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT must be set}"

export RECAPTION_CACHE_ROOT_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp/${USER}/recaption-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}}"
export RECAPTION_LOGICAL_TASK_COUNT=$((SLURM_ARRAY_TASK_COUNT * WORKERS_PER_NODE))

mkdir -p logs

echo "Array job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
echo "Started at $(date --iso-8601=seconds)"
echo "Loader ${RECAPTION_LOADER}"
echo "Workers per node ${WORKERS_PER_NODE}"
echo "Logical task count ${RECAPTION_LOGICAL_TASK_COUNT}"
echo "Cache root base ${RECAPTION_CACHE_ROOT_BASE}"
echo "Using uenv prgenv-gnu/24.11:v1 with modules view"

uenv run prgenv-gnu/24.11:v1 --view=modules -- bash -lc '
module load cuda gcc cmake ninja
nvcc --version
ninja --version
srun \
  --ntasks="$WORKERS_PER_NODE" \
  --ntasks-per-node="$WORKERS_PER_NODE" \
  --gpus-per-task=1 \
  --cpus-per-task="$CPUS_PER_WORKER" \
  --cpu-bind=cores \
  --gpu-bind=single:1 \
  --kill-on-bad-exit=1 \
  --label \
  bash -lc '"'"'
worker_rank=${SLURM_PROCID:?}
logical_task_id=$((SLURM_ARRAY_TASK_ID * WORKERS_PER_NODE + worker_rank))
cache_root="${RECAPTION_CACHE_ROOT_BASE}/worker-${worker_rank}"
xdg_cache_root="${cache_root}/xdg-cache"
vllm_cache_root="${cache_root}/vllm-cache"
vllm_rpc_base_path="${cache_root}/vllm-rpc"

export SLURM_ARRAY_TASK_ID="${logical_task_id}"
export SLURM_ARRAY_TASK_COUNT="${RECAPTION_LOGICAL_TASK_COUNT}"
export FLASHINFER_WORKSPACE_BASE="${cache_root}/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
export TRITON_CACHE_DIR="${cache_root}/triton"
export XDG_CACHE_HOME="${xdg_cache_root}"
export VLLM_CACHE_ROOT="${vllm_cache_root}"
export VLLM_RPC_BASE_PATH="${vllm_rpc_base_path}"
export VLLM_PORT=$((39000 + logical_task_id * 4))

mkdir -p \
  "${FLASHINFER_WORKSPACE_BASE}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${VLLM_CACHE_ROOT}" \
  "${VLLM_RPC_BASE_PATH}"

echo "Worker ${worker_rank}: logical task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
echo "Worker ${worker_rank}: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Worker ${worker_rank}: cache root ${cache_root}"
echo "Worker ${worker_rank}: VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
echo "Worker ${worker_rank}: VLLM_RPC_BASE_PATH=${VLLM_RPC_BASE_PATH}"
echo "Worker ${worker_rank}: VLLM_PORT=${VLLM_PORT}"

.venv/bin/python main.py
'"'"'
'

echo "Finished at $(date --iso-8601=seconds)"
