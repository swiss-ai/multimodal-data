#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --time=12:00:00
#SBATCH --job-name=ego4d
#SBATCH --array=0-2

DATASETS=(
  "full_scale annotations"
  "video_540ss annotations_540ss"
  "clips"
)
OUTPUT_DIRS=(
  "/path/to/data/vision-datasets/ego4d"
  "/path/to/data/vision-datasets/ego4d"
  "/path/to/data/vision-datasets/ego4d_clips"
)

echo "Starting job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Job name: $SLURM_JOB_NAME"

ego4d yes \
  --dataset "${DATASETS[$SLURM_ARRAY_TASK_ID]}" \
  --output_directory "${OUTPUT_DIRS[$SLURM_ARRAY_TASK_ID]}" \
  --version v2

echo "Finished job $SLURM_JOB_ID on $(hostname) at $(date)"
