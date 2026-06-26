#!/bin/bash
# Download i4ds/SRG_apertus_data (~819GB, 3552h German Swiss broadcast speech)
# Private dataset — requires HF_TOKEN. Token is temporary; will be revoked once stored.
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-srg-apertus
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/srg-apertus-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/srg-apertus-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive

set -euo pipefail

DATASET_REPO="i4ds/SRG_apertus_data"
REVISION="b59ff064e695d3b497a1041e5437f002b14dfaa8"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___i4ds___SRG_apertus_data"

export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

uv pip install --python /opt/venv/bin/python hf_transfer

echo "========================================"
echo "SRG Apertus Data download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --revision "${REVISION}" \
  --token "${HF_TOKEN}" \
  --local-dir "${DEST_DIR}" \
  --max-workers 64

echo "[$(date '+%F %T')] Download finished"

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "Done."
