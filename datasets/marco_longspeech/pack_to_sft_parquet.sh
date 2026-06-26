#!/bin/bash
# Pack AIDC-AI/Marco_Longspeech raw JSONL + WAVs into normalized audio-SFT Parquet.
#
# Run from a compute node:
#   bash pack_to_sft_parquet.sh

set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/tmp/pycache-${USER}-marco-sft-pack
export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"

SCRIPT_DIR="/iopsstor/scratch/cscs/${USER}/apertus/multimodal-data/01-dataset-download/audio/marco_longspeech"
RAW_ROOT="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/sft/hf___AIDC-AI___Marco_Longspeech"
OUTPUT_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/processed/sft/hf___AIDC-AI___Marco_Longspeech"

echo "========================================"
echo "Marco_Longspeech -> normalized audio-SFT Parquet"
echo "Raw:      ${RAW_ROOT}"
echo "Output:   ${OUTPUT_DIR}"
echo "Workers:  16"
echo "Media rows per shard: 512"
echo "Started:  $(date '+%F %T')"
echo "========================================"

python "${SCRIPT_DIR}/pack_to_sft_parquet.py" \
  --raw-root "${RAW_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --media-audio-rows-per-shard 512 \
  --example-rows-per-shard 50000 \
  --media-audio-write-batch-size 2 \
  --num-workers 16 \
  --overwrite

echo "[$(date '+%F %T')] Done"
du -sh "${OUTPUT_DIR}"
