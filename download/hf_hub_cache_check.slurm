#!/bin/bash
# Set your SLURM account: #SBATCH --account=<your-account>
#SBATCH --job-name=checkDset
#SBATCH --output=logs/hf-%x-%A.out
#SBATCH --error=logs/hf-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1

DATASET_NAME="${DATASET_NAME:-mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M}"
CLUSTER_REPO_HOME="${CLUSTER_REPO_HOME:-$SLURM_SUBMIT_DIR}"
NUM_PROC=128
BATCH_SIZE=10

mkdir -p "${CLUSTER_REPO_HOME}/01-dataset-download/logs"

pip install -q -r "${CLUSTER_REPO_HOME}/01-dataset-download/requirements.txt"

python "${CLUSTER_REPO_HOME}/01-dataset-download/hf_hub_cache_check.py" \
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
