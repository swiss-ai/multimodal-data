#!/bin/bash
# Download doof-ferb/infore2_audiobooks (~47GB, 415h Vietnamese audiobooks, 315k samples, CC BY 4.0)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-infore2
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/infore2-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/infore2-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=SD-69241-apertus-1-5-5

set -euo pipefail

DATASET_REPO="doof-ferb/infore2_audiobooks"
REVISION="836a9a79e28c6e598a5c308de7f047613dcf9ec5"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/doof-ferb___infore2_audiobooks"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "infore2_audiobooks download"
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
echo "Pinned to commit: ${REVISION}"

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "Done."
