#!/usr/bin/env bash
#SBATCH --job-name=smithsonian-clean-all
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=288
#SBATCH --array=0-3
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

VENV=.venv
SCRIPT_DIR=.
NUM_WORKERS=4   # keep in sync with --array=0-3

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Started at $(date --iso-8601=seconds)"
echo "Worker: ${SLURM_ARRAY_TASK_ID} / ${NUM_WORKERS}"
echo "Node: $(hostname)"

uenv run prgenv-gnu/24.11:v1 --view=modules -- bash -lc "
module load cuda gcc cmake ninja
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
${VENV}/bin/python ${SCRIPT_DIR}/03_clean_all.py \
    --worker-id ${SLURM_ARRAY_TASK_ID} \
    --num-workers ${NUM_WORKERS}
"

echo "Finished at $(date --iso-8601=seconds)"
echo "Output root: /path/to/data/vision-datasets/processed/smithsonian_cleaned"
