#!/bin/bash
# Download HuggingFaceM4/WebSight v0.2 (~1.92M rows, screenshots + HTML/Tailwind CSS)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-websight
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/websight-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/websight-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="HuggingFaceM4/WebSight"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/HuggingFaceM4___WebSight"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "WebSight v0.2 download"
echo "Repo: ${DATASET_REPO}"
echo "Dest: ${DEST_DIR}"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --include "v0.2/*" \
  --local-dir "${DEST_DIR}" \
  --max-workers 32

echo "[$(date '+%F %T')] Download finished"

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "Done."
