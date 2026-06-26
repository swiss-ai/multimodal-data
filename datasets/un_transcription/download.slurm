#!/bin/bash
# Download united-nations/transcription-corpus (~17GB, CC-BY-4.0)
# UN Security Council + General Assembly meetings, 6 languages (en/fr/es/ar/zh/ru)
# Parallel audio per language + official verbatim records.
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-un-corpus
#SBATCH --reservation=SD-69241-apertus-1-5-6
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/un-corpus-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/un-corpus-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288

set -euo pipefail

DATASET_REPO="united-nations/transcription-corpus"
REVISION="e55f7d9d2e2e7cac4e2f5c6b6c89cd03c87a5b62"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___united-nations___transcription-corpus"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "UN Transcription Corpus download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --revision "${REVISION}" \
  --local-dir "${DEST_DIR}" \
  --max-workers 64

echo "[$(date '+%F %T')] Download finished"

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "Done."
