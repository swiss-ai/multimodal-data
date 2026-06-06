#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

NUM_WORKERS="${NUM_WORKERS:-64}"
AUDIO_ROWS_PER_SHARD="${AUDIO_ROWS_PER_SHARD:-512}"

python "${SCRIPT_DIR}/pack_matching_sft_to_parquet.py" \
  --num-workers "${NUM_WORKERS}" \
  --audio-rows-per-shard "${AUDIO_ROWS_PER_SHARD}" \
  "$@"
