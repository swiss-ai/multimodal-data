#!/bin/bash
# Download FreedomIntelligence/TCM-Pretrain-Data-ShizhenGPT from HuggingFace
# ~228 GB, 5B+ tokens, Traditional Chinese Medicine text+image dataset
# 5 configs: TCM_Book_Corpus, TCM_Web_Corpus, TCM_Web_Interleaved_Data, TCM_Book_Interleaved_Data, TCM_Synthesized_Vision
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-tcmshizhen
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/tcmshizhen-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/tcmshizhen-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="FreedomIntelligence/TCM-Pretrain-Data-ShizhenGPT"
REVISION="db4874ce4e322f47432fe322c558e516c5aad71e"  # main as of 2025-09-08
DEST_DIR=${DEST_DIR:-/capstor/store/cscs/swissai/infra01/vision-datasets/raw/hf___FreedomIntelligence___TCM-Pretrain-Data-ShizhenGPT}

# Install hf_transfer for faster downloads
uv pip install --python /opt/venv/bin/python hf_transfer

# Use a temporary cache dir to avoid polluting home and enable easy cleanup
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "TCM-Pretrain-Data-ShizhenGPT Download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

mkdir -p "${DEST_DIR}"

huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --revision "${REVISION}" \
  --local-dir "${DEST_DIR}" \
  --max-workers 64

echo "[$(date '+%F %T')] Download finished"
echo "Pinned to commit: ${REVISION}"

# Clean up HF hub cache (duplicate of downloaded files)
echo "Cleaning up HF hub cache..."
rm -rf "${HF_HUB_CACHE}"
echo "Done."
