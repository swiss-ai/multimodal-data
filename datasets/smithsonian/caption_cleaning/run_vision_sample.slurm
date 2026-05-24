#!/usr/bin/env bash
#SBATCH --job-name=smithsonian-vision-sample
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=288
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

VENV=.venv
SCRIPT_DIR=.

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Started at $(date --iso-8601=seconds)"
echo "Node: $(hostname)"

uenv run prgenv-gnu/24.11:v1 --view=modules -- bash -lc "
module load cuda gcc cmake ninja
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
${VENV}/bin/python ${SCRIPT_DIR}/04_vision_sample.py
"

echo "Finished at $(date --iso-8601=seconds)"
