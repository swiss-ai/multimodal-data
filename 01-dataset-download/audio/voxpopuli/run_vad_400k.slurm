#!/bin/bash
# Run Silero VAD on VoxPopuli 400k — one job per language (23 array tasks).
# Each task VADs a single language from iopsstor with 288 workers.
#
# Usage:
#   sbatch run_vad_400k.slurm
#
#SBATCH --account=infra01
#SBATCH --environment=nemo
#SBATCH --reservation=PA-2338-RL
#SBATCH --job-name=vad-400k
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/vad400k-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/vad400k-%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --array=0-22

set -euo pipefail

# Install dependencies
uv pip install --system --break-system-packages --no-deps --no-build-isolation \
    git+https://github.com/pytorch/audio.git@release/2.9
uv pip install --system --break-system-packages --no-deps \
    onnxruntime speechbrain hyperpyyaml silero-vad orjson

LANGS=(bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro sk sl sv)
LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

AUDIO_DIR="/iopsstor/scratch/cscs/${USER}/audio-datasets/raw/voxpopuli/${LANG}"
OUTPUT_DIR="/iopsstor/scratch/cscs/${USER}/audio-datasets/raw/voxpopuli/vad_results"
NUM_WORKERS=288
DATASET="voxpopuli"

mkdir -p "${OUTPUT_DIR}"

if [ ! -d "${AUDIO_DIR}" ]; then
    echo "[${LANG}] Audio dir not found: ${AUDIO_DIR}"
    exit 1
fi

echo "=========================================="
echo "VoxPopuli VAD — ${LANG} (task ${SLURM_ARRAY_TASK_ID}/22)"
echo "Audio: ${AUDIO_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Workers: ${NUM_WORKERS}"
echo "=========================================="

cd /iopsstor/scratch/cscs/${USER}/apertus/benchmark-audio-tokenizer

python -m audio_tokenization.utils.prepare_data.run_vad \
    --audio_dir "${AUDIO_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --dataset "${DATASET}" \
    --num_workers "${NUM_WORKERS}" \
    --backend onnx \
    --pool_start spawn

echo "[$(date '+%F %T')] Done: ${LANG}"
