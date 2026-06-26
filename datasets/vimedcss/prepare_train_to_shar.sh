#!/usr/bin/env bash
# Convert ViMedCSS train split parquet shards -> SHAR (stage_2)

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}
LHOTSE_REPO=${LHOTSE_REPO:-/iopsstor/scratch/cscs/xyixuan/dev/lhotse}
BENCHMARK_REPO=${BENCHMARK_REPO:-/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer}

FFMPEG_ROOT=${FFMPEG_ROOT:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64}
FFMPEG_BIN=${FFMPEG_BIN:-${FFMPEG_ROOT}/bin/ffmpeg}
PARQUET_DIR=${PARQUET_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/tensorxt___ViMedCSS/data}
SHAR_OUT_DIR=${SHAR_OUT_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/vimedcss_train}
TEXT_TOKENIZER=${TEXT_TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json}

PARQUET_GLOB=${PARQUET_GLOB:-train-*.parquet}
NUM_WORKERS=${NUM_WORKERS:-30}
TARGET_SR=${TARGET_SR:-24000}
SHARD_SIZE=${SHARD_SIZE:-5000}

export PATH="${FFMPEG_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${LHOTSE_REPO}:${BENCHMARK_REPO}:${PYTHONPATH:-}"

if [[ ! -x "${FFMPEG_BIN}" ]]; then
  echo "ERROR: ffmpeg binary not found/executable: ${FFMPEG_BIN}" >&2
  echo "Hint: set FFMPEG_BIN=/abs/path/to/ffmpeg or FFMPEG_ROOT=/abs/path/to/ffmpeg-runtime" >&2
  exit 1
fi

echo "=========================================="
echo "ViMedCSS train -> SHAR"
echo "Parquet dir:   ${PARQUET_DIR}"
echo "Parquet glob:  ${PARQUET_GLOB}"
echo "Output dir:    ${SHAR_OUT_DIR}"
echo "Workers:       ${NUM_WORKERS}"
echo "Target SR:     ${TARGET_SR}"
echo "Shard size:    ${SHARD_SIZE}"
echo "Python:        ${PYTHON_BIN}"
echo "FFmpeg root:   ${FFMPEG_ROOT}"
echo "FFmpeg bin:    ${FFMPEG_BIN}"
echo "=========================================="

"${FFMPEG_BIN}" -version | sed -n '1p'

"${PYTHON_BIN}" -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
  --parquet-dir "${PARQUET_DIR}" \
  --parquet-glob "${PARQUET_GLOB}" \
  --shar-dir "${SHAR_OUT_DIR}" \
  --id-column segment_id \
  --text-column segment_text \
  --target-sr "${TARGET_SR}" \
  --text-tokenizer "${TEXT_TOKENIZER}" \
  --shard-size "${SHARD_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --ffmpeg-bin "${FFMPEG_BIN}"

echo "Done: ${SHAR_OUT_DIR}"
