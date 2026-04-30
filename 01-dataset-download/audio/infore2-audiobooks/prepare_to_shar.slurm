#!/bin/bash
# Convert doof-ferb/infore2_audiobooks parquets to SHAR (stage_2).
# 111 parquets, ~44GB, ~315K segments, Vietnamese audiobooks.
# Audio paths encode book/chapter + segment: ANhemsairoitap1c17_022.wav
# -> source_id="ANhemsairoitap1c17", clip_num=22 via trailing_number parser.
# Consecutive clip_nums within a book/chapter can be interleaved.
#
# Usage:
#   sbatch prepare_to_shar.slurm

#SBATCH --account=infra01
#SBATCH --job-name=infore2-to-shar
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/infore2-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/infore2-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}
LHOTSE_REPO=${LHOTSE_REPO:-/iopsstor/scratch/cscs/xyixuan/dev/lhotse}
BENCHMARK_REPO=${BENCHMARK_REPO:-/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer}

RAW_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/doof-ferb___infore2_audiobooks/data"
SHAR_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/infore2_audiobooks_train"

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
echo "Infore2 Audiobooks -> SHAR (stage_2)"
echo "Raw:       ${RAW_DIR}"
echo "SHAR:      ${SHAR_DIR}"
echo "Workers:   ${NUM_WORKERS}"
echo "Target SR: ${TARGET_SR}"
echo "=========================================="

"${PYTHON_BIN}" -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${RAW_DIR}" \
    --parquet-glob "train-*.parquet" \
    --shar-dir "${SHAR_DIR}" \
    --text-column transcription \
    --id-column audio.path \
    --language vi \
    --input-clip-id-parser trailing_number_basename \
    --target-sr "${TARGET_SR}" \
    --text-tokenizer "${TEXT_TOKENIZER}" \
    --shard-size "${SHARD_SIZE}" \
    --num-workers "${NUM_WORKERS}"

echo "[$(date '+%F %T')] Done"
