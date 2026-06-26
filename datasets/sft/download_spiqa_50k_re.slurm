#!/bin/bash
# Download scimdr/SPIQA_50K_Re (~138MB annotations) +
# google/spiqa SPIQA_train_val_Images.zip (~32GB images)
#SBATCH --account=infra01
#SBATCH --job-name=dl-spiqa_50k
#SBATCH --reservation=SD-69241-apertus-1-5-5
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/spiqa_50k_re_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/spiqa_50k_re_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DEST="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___scimdr___SPIQA_50K_Re"
mkdir -p "${DEST}"
export HF_HUB_CACHE="${DEST}/.hf_cache"

# ---- 1. Annotations from scimdr/SPIQA_50K_Re ----
DATASET_REPO="scimdr/SPIQA_50K_Re"
REVISION="964e2d260130"

echo "=========================================="
echo "[1/2] Downloading annotations: ${DATASET_REPO}"
echo "=========================================="

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${DEST}" \
    --max-workers 64

# ---- 2. Images from google/spiqa ----
IMAGES_REPO="google/spiqa"
IMAGES_REVISION="1774b71511f0"

echo "=========================================="
echo "[2/2] Downloading images: ${IMAGES_REPO}/train_val/SPIQA_train_val_Images.zip (~32GB)"
echo "=========================================="

huggingface-cli download "${IMAGES_REPO}" \
    train_val/SPIQA_train_val_Images.zip \
    --repo-type dataset \
    --revision "${IMAGES_REVISION}" \
    --local-dir "${DEST}" \
    --max-workers 64

rm -rf "${HF_HUB_CACHE}"
echo "Done: spiqa_50k_re"
du -sh "${DEST}"
