#!/bin/bash
# Download hostbot77/news_youtube_uzbek_speech_dataset (Uzbek speech, CC-BY-4.0)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-uzbek
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/download/uzbek-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/download/uzbek-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="hostbot77/news_youtube_uzbek_speech_dataset"
REVISION="16338402f9d1431cf5cca6912f4d0a1523bff356"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___hostbot77___news_youtube_uzbek_speech_dataset"

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

echo "========================================"
echo "Uzbek Speech Download"
echo "Repo: ${DATASET_REPO}, Dest: ${DEST_DIR}"
echo "========================================"

mkdir -p "${DEST_DIR}"

huggingface-cli download "${DATASET_REPO}" --repo-type dataset --revision "${REVISION}" --local-dir "${DEST_DIR}" --max-workers 64

echo "[$(date '+%F %T')] Download finished"
rm -rf "${HF_HUB_CACHE}"
du -sh "${DEST_DIR}"
