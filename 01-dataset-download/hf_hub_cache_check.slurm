#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --environment=emu3
#SBATCH --job-name=checkDset
#SBATCH --output=/iopsstor/scratch/cscs/%u/multimodal-data/01-dataset-download/logs/hf-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/multimodal-data/01-dataset-download/logs/hf-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1

# Specify root dir to look for parquet files (Huggingface hub cache for example)
DATASET_NAME="${DATASET_NAME:-mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M}"
NUM_PROC=128
BATCH_SIZE=10

python /iopsstor/scratch/cscs/$USER/multimodal-data/01-dataset-download/mproc_hub_cache_check.py \
                        --dataset="$DATASET_NAME" \
                        --workers=$NUM_PROC \
                        --batch-size=$BATCH_SIZE

EXIT_CODE=$?

# ========================================
# Job Summary
# ========================================

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
