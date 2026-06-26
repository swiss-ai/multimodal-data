#!/bin/bash
# Convert Omnilingual ASR to SHAR (stage_2).
# 348 languages, ~3Kh, parquet with embedded audio.
# Prompt (English question) stored in custom, raw_text as supervision.
#
# Usage:
#   sbatch prepare_to_shar.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=omni-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/omnilingual_asr_shar_%A.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/omnilingual_asr_shar_%A.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
RUNTIME_ENV_SH="${REPO_DIR}/scripts/source_lhotse_runtime.sh"
TEXT_TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

DATA_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___facebook___omnilingual-asr-corpus/data
SHAR_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/omnilingual_asr

cd "${REPO_DIR}"
source "${RUNTIME_ENV_SH}"
uv pip install --python /opt/venv/bin/python orjson polars zstandard

echo "=========================================="
echo "Omnilingual ASR → SHAR"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
  --parquet-dir "${DATA_ROOT}" \
  --parquet-glob "*/train-*.parquet" \
  --shar-dir "${SHAR_DIR}" \
  --id-column language speaker_id prompt_id segment_id \
  --text-column raw_text \
  --audio-column audio \
  --language-column language \
  --custom-columns prompt language speaker_id segment_id \
  --text-tokenizer "${TEXT_TOKENIZER}" \
  --target-sr 24000 \
  --shard-size 20000 \
  --num-workers 188

echo "Done"
