#!/bin/bash
# Download NeurologyAI/neuro-parakeet-food from HuggingFace (gated, auto-approval)
# NOTE: Requires HF token with accepted gated access
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-neuroparakeet
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/neuroparakeet-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/neuroparakeet-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="NeurologyAI/neuro-parakeet-food"
REVISION="5c9a5ab1e3e8945ba333a7509146e71448b1f6af"  # main as of 2026-03-28
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___NeurologyAI___neuro-parakeet-food"

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

echo "========================================"
echo "neuro-parakeet-food Download"
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

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "Done."
