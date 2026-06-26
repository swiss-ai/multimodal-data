#!/bin/bash
# Download JavisVerse/JavisData-audios from HuggingFace
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-javisdata
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/javisdata-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/javisdata-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="JavisVerse/JavisData-audios"
REVISION="8dc1e45deaba07de41a2b8f24d1d03b832d87220"  # main as of 2026-02-23
DEST_DIR=${DEST_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/javisdata-audios}

# Use a temporary cache dir to avoid polluting home and enable easy cleanup
uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "JavisData-audios Download"
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
