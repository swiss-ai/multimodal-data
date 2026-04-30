#!/bin/bash
# Convert otoSpeech full-duplex WDS tars to Lhotse SHAR.
# 61 tar shards, 538 sessions, 141h English conversational speech (stereo, 44.1kHz).
# Audio-only (no transcription in metadata).
#
# Usage:
#   sbatch prepare_to_shar.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=oto-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/otospeech_shar_%A.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/otospeech_shar_%A.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
LHOTSE_DIR=/iopsstor/scratch/cscs/xyixuan/dev/lhotse
FFMPEG_ROOT=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64
TORCHCODEC_WHL=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/torchcodec-0.9.0-cp312-cp312-linux_aarch64.whl

DATA_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___otoearth___otoSpeech-full-duplex-processed-141h
WDS_DIR="${DATA_ROOT}/data/train"
SHAR_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/otospeech

NUM_WORKERS=61
SHARD_SIZE=600
TARGET_SR=24000
MIN_SR=16000

HOST="$(hostname -s)"
mkdir -p "${SHAR_DIR}"

echo "=========================================="
echo "otoSpeech WDS → SHAR"
echo "=========================================="
echo "Host: ${HOST}, CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Input: ${WDS_DIR}"
echo "Output: ${SHAR_DIR}"
echo "Workers: ${NUM_WORKERS}"
echo "=========================================="

cd "${REPO_DIR}"
export PYTHONPATH="${LHOTSE_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
export PATH="${FFMPEG_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}"
uv pip install --python /opt/venv/bin/python --no-deps --force-reinstall "${TORCHCODEC_WHL}"

python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
  --wds-shards "${WDS_DIR}"/*.tar \
  --shar-dir "${SHAR_DIR}" \
  --num-workers "${NUM_WORKERS}" \
  --shard-size "${SHARD_SIZE}" \
  --target-sr "${TARGET_SR}" \
  --min-sr "${MIN_SR}" \
  --shar-format flac \
  --vad-segmentation \
  --vad-per-shard-dir "${DATA_ROOT}/vad_per_shard" \
  --vad-max-merge-gap-sec 0.5 \
  --vad-min-chunk-sec 0 \
  --vad-max-chunk-sec 200 \
  --vad-max-duration-sec 3600

echo "otoSpeech SHAR conversion finished"
