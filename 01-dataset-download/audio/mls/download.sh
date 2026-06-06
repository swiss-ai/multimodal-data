#!/bin/bash
# Download Multilingual LibriSpeech (facebook/multilingual_librispeech) — 7 non-English
# subsets in parquet format (~88 GiB) for audio tokenization.
#
# Subsets: german, dutch, french, spanish, italian, portuguese, polish
# (each top-level dir holds the train/dev/test/9_hours/1_hours parquet shards;
#  the repo's legacy `data/` audio tree is intentionally NOT downloaded.)
#
# Usage:
#   sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/mls/download.sh
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-mls
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mls-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mls-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=SD-69241-apertus-1-5-0

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_REPO="facebook/multilingual_librispeech"
REVISION="2e83e61823b4c47dcbcb1980bb88601274127609"  # main as of 2026-06-06
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___facebook___multilingual_librispeech"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"
mkdir -p "${DEST_DIR}"

echo "========================================"
echo "Multilingual LibriSpeech (7 non-English subsets) Download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

# All 7 language patterns go in a SINGLE --include (multiple --include flags
# silently overwrite each other in huggingface-cli).
huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --revision "${REVISION}" \
  --local-dir "${DEST_DIR}" \
  --include "german/*" "dutch/*" "french/*" "spanish/*" "italian/*" "portuguese/*" "polish/*" \
  --max-workers 64

echo "[$(date '+%F %T')] Download finished"
rm -rf "${HF_HUB_CACHE}"
echo "Final size:"
du -sh "${DEST_DIR}"
echo "Done."
