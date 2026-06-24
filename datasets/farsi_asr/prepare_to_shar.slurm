#!/bin/bash
# Convert farsi-asr/farsi-asr-dataset parquets to SHAR (stage_2).
# 331 parquet shards, ~300K Persian/Farsi ASR segments.
# Audio paths encode source + segment ordering inside audio.path.
# Example: Amirkaye_ziba_Goftegoo/Amrikaye_Ziba_radio-goftego-99_04_12-19_30_86.wav
# -> source_id="Amrikaye_Ziba_radio-goftego-99_04_12-19_30", clip_num=86
# via trailing_number_basename parser.

#SBATCH --account=infra01
#SBATCH --job-name=farsi-asr-to-shar
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/farsi-asr-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/farsi-asr-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}
LHOTSE_REPO=${LHOTSE_REPO:-/iopsstor/scratch/cscs/xyixuan/dev/lhotse}
BENCHMARK_REPO=${BENCHMARK_REPO:-/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer}

RAW_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___farsi-asr___farsi-asr-dataset/parquet"
SHAR_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/farsi_asr_train"

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
echo "Farsi ASR -> SHAR (stage_2)"
echo "Raw:       ${RAW_DIR}"
echo "SHAR:      ${SHAR_DIR}"
echo "Workers:   ${NUM_WORKERS}"
echo "Target SR: ${TARGET_SR}"
echo "=========================================="

"${PYTHON_BIN}" -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${RAW_DIR}" \
    --shar-dir "${SHAR_DIR}" \
    --text-column transcription \
    --id-column audio.path \
    --language fa \
    --input-clip-id-parser trailing_number_basename \
    --target-sr "${TARGET_SR}" \
    --text-tokenizer "${TEXT_TOKENIZER}" \
    --shard-size "${SHARD_SIZE}" \
    --num-workers "${NUM_WORKERS}"

echo "[$(date '+%F %T')] Done"
