#!/bin/bash
# Download tensorxt/ViMedCSS (~15.8K utterances, 32.6h Vietnamese medical code-switching ASR, 18.3GB, CC BY 4.0)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-vimedcss
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/vimedcss-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/vimedcss-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="tensorxt/ViMedCSS"
REVISION="b6959a18a08739464733930a872e7125c03e6558"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/tensorxt___ViMedCSS"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "ViMedCSS download"
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
