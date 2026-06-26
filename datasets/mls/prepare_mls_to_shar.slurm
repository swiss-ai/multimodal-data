#!/bin/bash
# Convert MLS (Multilingual LibriSpeech) parquets to SHAR with full metadata.
# 7 non-English languages, ~44Kh audiobook readings.
# Skip English — overlaps with LibriHeavy (same LibriVox source).
# Includes: speaker_id, chapter_id, begin_time, end_time, language.
# Interleave-ready: source_id from trailing_number parser.
#
# Usage:
#   sbatch --array=0-6 prepare_mls_to_shar.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=mls-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/mls_shar_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/mls_shar_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
RUNTIME_ENV_SH="${REPO_DIR}/scripts/utils/source_lhotse_runtime.sh"
TEXT_TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

RAW_BASE=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___facebook___multilingual_librispeech
SHAR_BASE=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/mls_v2

cd "${REPO_DIR}"
source "${RUNTIME_ENV_SH}"
uv pip install --python /opt/venv/bin/python orjson

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert

LANGS=(dutch french german italian polish portuguese spanish)
LANG_CODES=(nl fr de it pl pt es)
lang="${LANGS[${SLURM_ARRAY_TASK_ID}]}"
lang_code="${LANG_CODES[${SLURM_ARRAY_TASK_ID}]}"
SHAR_DIR="${SHAR_BASE}/mls_${lang}_train"
PARQUET_DIR="${RAW_BASE}/${lang}"

if [ -f "${SHAR_DIR}/_SUCCESS" ]; then
    echo "[mls_${lang}] done, skipping"
    exit 0
fi

echo "=========================================="
echo "MLS ${lang} (${lang_code}) → SHAR (task ${SLURM_ARRAY_TASK_ID})"
echo "Parquet dir: ${PARQUET_DIR}"
echo "Output: ${SHAR_DIR}"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${PARQUET_DIR}" \
    --parquet-glob "train-*.parquet" \
    --shar-dir "${SHAR_DIR}" \
    --id-column id \
    --text-column transcript \
    --audio-column audio \
    --custom-columns speaker_id chapter_id begin_time end_time audio_duration \
    --text-tokenizer "${TEXT_TOKENIZER}" \
    --target-sr 24000 \
    --shard-size 20000 \
    --input-clip-id-parser trailing_number \
    --language "${lang_code}" \
    --num-workers 128

echo "Done: mls_${lang}"
