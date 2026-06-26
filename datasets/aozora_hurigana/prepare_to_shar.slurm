#!/bin/bash
# Convert Aozora Hurigana Speech v2 parquet shards to SHAR (cooldown).
# Parquet rows preserve rendition-level sequencing:
#   sample_id = "{author}__{work}__{rendition}_{clip_num:06d}"
#   source_id = "{author}__{work}__{rendition}"
#   clip_num  = line_num
# This is the safe interleave granularity for the dataset.

#SBATCH --account=infra01
#SBATCH --job-name=aozora-to-shar
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/aozora-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/aozora-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}
LHOTSE_REPO=${LHOTSE_REPO:-/iopsstor/scratch/cscs/xyixuan/dev/lhotse}
BENCHMARK_REPO=${BENCHMARK_REPO:-/iopsstor/scratch/cscs/xyixuan/dev/benchmark-audio-tokenizer-interleave-v2}

RAW_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/ndl___aozora_hurigana_speech_v2/parquet"
SHAR_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/cooldown/aozora_hurigana_train"

FFMPEG_ROOT=${FFMPEG_ROOT:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64}
TEXT_TOKENIZER=${TEXT_TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json}

NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-288}}
TARGET_SR=${TARGET_SR:-24000}
SHARD_SIZE=${SHARD_SIZE:-5000}

export PATH="${FFMPEG_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${LHOTSE_REPO}:${BENCHMARK_REPO}:${PYTHONPATH:-}"
uv pip install --python /opt/venv/bin/python polars

echo "=========================================="
echo "Aozora Hurigana -> SHAR (cooldown)"
echo "Parquet:   ${RAW_DIR}"
echo "SHAR:      ${SHAR_DIR}"
echo "Repo:      ${BENCHMARK_REPO}"
echo "Workers:   ${NUM_WORKERS}"
echo "Target SR: ${TARGET_SR}"
echo "=========================================="

"${PYTHON_BIN}" -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${RAW_DIR}" \
    --parquet-glob "train-*.parquet" \
    --shar-dir "${SHAR_DIR}" \
    --text-column transcription \
    --id-column sample_id \
    --language ja \
    --custom-columns author work rendition line_num \
    --input-clip-id-parser trailing_number \
    --target-sr "${TARGET_SR}" \
    --text-tokenizer "${TEXT_TOKENIZER}" \
    --shard-size "${SHARD_SIZE}" \
    --num-workers "${NUM_WORKERS}"

echo "[$(date '+%F %T')] Done"
