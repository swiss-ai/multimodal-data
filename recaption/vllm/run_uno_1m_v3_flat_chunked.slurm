#!/usr/bin/env bash
#SBATCH --job-name=uno-1m-v3-flat-chunk
#SBATCH --array=0-99
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --reservation=PA-2338-RL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

mkdir -p logs

echo "Array job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
echo "Started at $(date --iso-8601=seconds)"

python \
  scripts/consolidate_uno_1m_v3_flat_chunk.py \
  --chunk-id "${SLURM_ARRAY_TASK_ID}" \
  --chunk-count "${SLURM_ARRAY_TASK_COUNT}"

echo "Finished at $(date --iso-8601=seconds)"
