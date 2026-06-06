#!/bin/bash
# Download jmhessel/newyorker_caption_contest (~7.4 GB, 100 parquet files,
# 30 configs spanning matching/ranking/explanation tasks, image-to-text /
# VQA / multiple-choice).
#
# Usage:
#   sbatch download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-nycaption
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/nycaption-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/nycaption-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288

set -euo pipefail

DATASET_REPO="jmhessel/newyorker_caption_contest"
REVISION="d81cbab7d0392708d5371d3a4960e69261824db4"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/hf___jmhessel___newyorker_caption_contest"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "New Yorker Caption Contest Download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

mkdir -p "${DEST_DIR}"

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
