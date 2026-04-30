#!/bin/bash
# Download uhhlt/Tuda-De (~17GB, curated German speech, CC-BY-4.0)
# From U-Hamburg/TU-Darmstadt. Wikipedia + Europarl readings, controlled studio.
# Train/test/dev splits — only use train for tokenization.
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-tuda-de
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/tuda-de-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/tuda-de-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=SD-69241-apertus-1-5-5

set -euo pipefail

DATASET_REPO="uhhlt/Tuda-De"
REVISION="26ad28a9ac73b664c2f7c7c93969f9507ea6aadc"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___uhhlt___Tuda-De"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "Tuda-De download"
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
