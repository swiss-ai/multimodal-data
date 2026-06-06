#!/bin/bash
# Download mtg-jamendo dataset from HuggingFace with pinned revision
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-mtg-jamendo
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mtg-jamendo-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mtg-jamendo-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="rkstgr/mtg-jamendo"
REVISION="8265518f6b5127d386a85ab5c380d867ff9ae70b"  # main as of 2026-02-20
DEST_DIR=${DEST_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/mtg-jamendo}

# Use a temporary cache dir to avoid polluting home and enable easy cleanup
uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "MTG-Jamendo Download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --revision "${REVISION}" \
  --local-dir "${DEST_DIR}"

echo "[$(date '+%F %T')] Download finished"
echo "Pinned to commit: ${REVISION}"

# Clean up HF hub cache (duplicate of downloaded files)
echo "Cleaning up HF hub cache..."
rm -rf "${HF_HUB_CACHE}"
echo "Done."
