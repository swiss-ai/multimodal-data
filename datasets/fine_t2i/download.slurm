#!/bin/bash
# Download ma-xu/fine-t2i from HuggingFace — one subset per array task
# 6M image-text pairs (~1.6TB), Apache 2.0
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-fine-t2i
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/fine-t2i-%x-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/fine-t2i-%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL
#SBATCH --array=0-4

set -euo pipefail

DATASET_REPO="ma-xu/fine-t2i"
REVISION="28fdd5663ee202b5cafc01d6ed08a03f14957854"  # main as of 2026-03-28
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/hf___ma-xu___fine-t2i"

SUBSETS=(
    curated
    synthetic_enhanced_prompt_random_resolution
    synthetic_enhanced_prompt_square_resolution
    synthetic_original_prompt_random_resolution
    synthetic_original_prompt_square_resolution
)
SUBSET="${SUBSETS[$SLURM_ARRAY_TASK_ID]}"

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache_${SUBSET}"

echo "========================================"
echo "fine-t2i download: ${SUBSET} (task ${SLURM_ARRAY_TASK_ID})"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

mkdir -p "${DEST_DIR}"

huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --revision "${REVISION}" \
  --local-dir "${DEST_DIR}" \
  --include "${SUBSET}/*" \
  --max-workers 64

echo "[$(date '+%F %T')] Download finished for ${SUBSET}"
echo "Pinned to commit: ${REVISION}"

rm -rf "${HF_HUB_CACHE}"
echo "Done."
