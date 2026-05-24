#!/usr/bin/env bash
#SBATCH --job-name=recaption-explicit-tasks
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
: "${RECAPTION_LOADER:?RECAPTION_LOADER must be set}"
: "${RECAPTION_EXPLICIT_TASK_IDS:?RECAPTION_EXPLICIT_TASK_IDS must be set}"
: "${RECAPTION_LOGICAL_TASK_COUNT:?RECAPTION_LOGICAL_TASK_COUNT must be set}"

TASK_IDS_RAW="${RECAPTION_EXPLICIT_TASK_IDS//:/,}"
IFS=, read -r -a EXPLICIT_TASK_IDS <<< "${TASK_IDS_RAW}"
WORKERS_PER_NODE="${#EXPLICIT_TASK_IDS[@]}"
if [[ "${WORKERS_PER_NODE}" -lt 1 ]]; then
  echo "No explicit logical task ids provided" >&2
  exit 1
fi
CPUS_PER_WORKER=$(( SLURM_CPUS_PER_TASK / WORKERS_PER_NODE ))

export RECAPTION_CACHE_ROOT_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp/${USER}/recaption-${SLURM_JOB_ID}}}"

mkdir -p logs

echo "Job ${SLURM_JOB_ID}"
echo "Started at $(date --iso-8601=seconds)"
echo "Loader ${RECAPTION_LOADER}"
echo "Explicit logical task ids ${RECAPTION_EXPLICIT_TASK_IDS}"
echo "Logical task count ${RECAPTION_LOGICAL_TASK_COUNT}"
echo "Workers per node ${WORKERS_PER_NODE}"
echo "CPUs per worker ${CPUS_PER_WORKER}"
echo "Cache root base ${RECAPTION_CACHE_ROOT_BASE}"
echo "Using uenv prgenv-gnu/24.11:v1 with modules view"

uenv run prgenv-gnu/24.11:v1 --view=modules -- bash -lc '
module load cuda gcc cmake ninja
nvcc --version
ninja --version
srun \
  --ntasks="'"${WORKERS_PER_NODE}"'" \
  --ntasks-per-node="'"${WORKERS_PER_NODE}"'" \
  --gpus-per-task=1 \
  --cpus-per-task="'"${CPUS_PER_WORKER}"'" \
  --cpu-bind=cores \
  --gpu-bind=single:1 \
  --kill-on-bad-exit=1 \
  --label \
  bash -lc '"'"'
worker_rank=${SLURM_PROCID:?}
task_ids_raw="${RECAPTION_EXPLICIT_TASK_IDS//:/,}"
IFS=, read -r -a explicit_task_ids <<< "${task_ids_raw}"
logical_task_id="${explicit_task_ids[$worker_rank]:?missing logical task id for worker}"
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
