#!/bin/bash
# Download OleehyO/latex-formulas-80M — 80M LaTeX formula renderings paired
# with their LaTeX source (en + zh + handwritten + benchmarks).  Apache-2.0.
#
# Re-download because the previously-mounted copy at
# /capstor/store/cscs/swissai/infra01/vision-datasets/raw/stage2/hf___OleehyO___latex-formulas-80M
# is broken: parquet shards are dangling symlinks into a missing blobs dir.
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-latex80m
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/latex80m-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/latex80m-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=SD-69241-apertus-1-5

set -euo pipefail

DATASET_REPO="OleehyO/latex-formulas-80M"
REVISION="5cd783320b0092caa85720a85d86595f42df043b"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/stage2/hf___OleehyO___latex-formulas-80M_v2"

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

echo "========================================"
echo "latex-formulas-80M re-download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "Files:    2693 parquet shards (en + zh + handwritten + benchmarks)"
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
echo
echo "Top-level layout:"
ls -la "${DEST_DIR}" | head -20
echo
echo "Done."
