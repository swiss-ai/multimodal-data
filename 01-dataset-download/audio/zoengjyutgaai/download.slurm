#!/bin/bash
# Download CanCLID/zoengjyutgaai (188h Cantonese storytelling, CC0)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-cantonese
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/cantonese-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/cantonese-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="CanCLID/zoengjyutgaai"
REVISION="829b30725811cabe9b8e2fd0513966470b304f12"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___CanCLID___zoengjyutgaai"

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

echo "========================================"
echo "Cantonese Storytelling Download"
echo "Repo: ${DATASET_REPO}, Dest: ${DEST_DIR}"
echo "========================================"

mkdir -p "${DEST_DIR}"

huggingface-cli download "${DATASET_REPO}" --repo-type dataset --revision "${REVISION}" --local-dir "${DEST_DIR}" --max-workers 64

echo "[$(date '+%F %T')] Download finished"
rm -rf "${HF_HUB_CACHE}"
du -sh "${DEST_DIR}"
