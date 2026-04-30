#!/bin/bash
# Download ghananlpcommunity/ghana-english-speech-600hrs (~105GB, 620h West African English, CC-BY-4.0)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-ghana-en
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/ghana-en-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/ghana-en-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=SD-69241-apertus-1-5-6

set -euo pipefail

DATASET_REPO="ghananlpcommunity/ghana-english-speech-600hrs"
REVISION="9c0b9da2d02cfea61c082582f2c628e39aebeeca"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___ghananlpcommunity___ghana-english-speech-600hrs"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "Ghana English 600hrs download"
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
