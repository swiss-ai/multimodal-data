#!/bin/bash
# Download MLS (Multilingual LibriSpeech) 7 non-English languages as parquet.
# Skip English — overlaps with LibriHeavy (same LibriVox source).
# Few large parquet files → use huggingface-cli with HF_HUB_ENABLE_HF_TRANSFER=1.
#
# Usage:
#   sbatch --array=0-6 download_mls.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dl-mls
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/mls_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/mls_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

DATASET_REPO="facebook/multilingual_librispeech"
REVISION="2e83e61823b4c47dcbcb1980bb88601274127609"  # latest as of 2024-08-12
DEST_BASE="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___facebook___multilingual_librispeech"

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_BASE}/.hf_cache"

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download

LANGS=(dutch french german italian polish portuguese spanish)
lang="${LANGS[${SLURM_ARRAY_TASK_ID}]}"

echo "=========================================="
echo "Downloading MLS: ${lang}"
echo "Repo: ${DATASET_REPO}@${REVISION}"
echo "Dest: ${DEST_BASE}"
echo "=========================================="

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --include "${lang}/train-*.parquet" \
    --local-dir "${DEST_BASE}" \
    --max-workers 64

rm -rf "${HF_HUB_CACHE}"
echo "Done: ${lang}"
