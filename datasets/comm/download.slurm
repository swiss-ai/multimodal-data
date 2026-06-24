#!/bin/bash
# Download weisuxi/CoMM -- train split only (arXiv:2406.10462, CC-BY-4.0).
# Just train_data.pth (~1.6 GB) + README. Train images are sourced externally
# via URLs inside the .pth annotations; only val_and_test_images.zip is bundled
# in the repo (skipped here).
#
# Usage:
#   sbatch download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-comm-train
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/comm-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/comm-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288

set -euo pipefail

DATASET_REPO="weisuxi/CoMM"
REVISION="029065b01f747578e9ef3acd47b14cc7c08d65bb"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/hf___weisuxi___CoMM"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "CoMM Download (train split only)"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

mkdir -p "${DEST_DIR}"

huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --revision "${REVISION}" \
  --local-dir "${DEST_DIR}" \
  --include "train_data.pth" "README.md" ".gitattributes" \
  --max-workers 64

echo "[$(date '+%F %T')] Download finished"

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "Done."
