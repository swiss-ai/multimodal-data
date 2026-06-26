#!/bin/bash
# Download allenai/OLMoASR-Pool (YouTube video IDs, ~17M rows, ~160MB)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-olmoasr-pool
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/olmoasr-pool-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/olmoasr-pool-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

set -euo pipefail

DATASET_REPO="allenai/OLMoASR-Pool"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/allenai___OLMoASR-Pool"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "OLMoASR-Pool download"
echo "Repo: ${DATASET_REPO}"
echo "Dest: ${DEST_DIR}"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --local-dir "${DEST_DIR}" \
  --max-workers 16

echo "[$(date '+%F %T')] Download finished"

# Clean up HF hub cache
rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "Done."
