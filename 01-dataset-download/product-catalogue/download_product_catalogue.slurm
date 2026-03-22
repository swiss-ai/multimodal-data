#!/bin/bash
# Download Shopify/product-catalogue (~48K rows, product images + taxonomy)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-product-catalogue
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/product-catalogue-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/product-catalogue-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="Shopify/product-catalogue"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/Shopify___product-catalogue"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "Shopify/product-catalogue download"
echo "Repo: ${DATASET_REPO}"
echo "Dest: ${DEST_DIR}"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --local-dir "${DEST_DIR}" \
  --max-workers 32

echo "[$(date '+%F %T')] Download finished"

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "Done."
