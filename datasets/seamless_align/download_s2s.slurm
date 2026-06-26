#!/bin/bash
# Download jhu-clsp/seamless-align S2S (Speech-to-Speech) subset.
# 33 language pairs, ~500GB compressed, WebDataset tar.gz with MP3 audio.
#
# Usage:
#   sbatch download_s2s.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dl-seamless-s2s
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/seamless_s2s_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/seamless_s2s_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

DATASET_REPO="jhu-clsp/seamless-align"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/jhu-clsp___seamless-align"
mkdir -p "${DEST_DIR}"
mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

echo "=========================================="
echo "Downloading jhu-clsp/seamless-align S2S"
echo "=========================================="
echo "Repo: ${DATASET_REPO}"
echo "Dest: ${DEST_DIR}"
echo "=========================================="

REVISION="8abc0ac98aedc10899e73a2d9e3c2c57cc6504fc"  # latest as of 2024-06-02

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --include "data_s2s/*" \
    --local-dir "${DEST_DIR}" \
    --max-workers 64

echo "=========================================="
echo "Download complete"
echo "=========================================="
