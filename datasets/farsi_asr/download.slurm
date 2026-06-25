#!/bin/bash
# Download farsi-asr/farsi-asr-dataset (~101GB, Persian/Farsi ASR, MIT)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-farsi-asr
#SBATCH --reservation=SD-69241-apertus-1-5-6
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/farsi-asr-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/farsi-asr-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288

set -euo pipefail

DATASET_REPO="farsi-asr/farsi-asr-dataset"
REVISION="59f14ee2ffce2f9e454d80f2b6d29016dd513492"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___farsi-asr___farsi-asr-dataset"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "Farsi ASR download"
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
