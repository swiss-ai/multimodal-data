#!/bin/bash
# Download eko57/spc_r_segmented from HuggingFace
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-spc-r-seg
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/spc-r-seg-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/spc-r-seg-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="eko57/spc_r_segmented"
REVISION="f63b9a302f0a29e55ed52397f806026887c7f71d"  # main as of 2026-02-24
DEST_DIR=${DEST_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/spc-r-segmented}

# Use a temporary cache dir to avoid polluting home and enable easy cleanup
uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "spc_r_segmented Download"
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
