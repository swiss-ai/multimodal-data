#!/bin/bash
# Build interleaved indexed datasets for VoxPopuli ASR, one language at a time.
# Uses by-lang split to avoid OOM on the full 46.7M clip dataset.
#
# Usage:
#   sbatch --array=0-22 build_voxpopuli_interleave.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=build-vp
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/build/voxpopuli_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/build/voxpopuli_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
uv pip install --python /opt/venv/bin/python orjson polars

CACHE_BASE=/capstor/store/cscs/swissai/infra01/audio-datasets/tokenized/interleave_cache/voxpopuli_asr_interleave_by_lang
OUT_BASE=/capstor/store/cscs/swissai/infra01/audio-datasets/tokenized/interleave/voxpopuli_asr
TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/build

mapfile -t LANGS < <(ls "${CACHE_BASE}" | sort)
lang="${LANGS[${SLURM_ARRAY_TASK_ID}]}"

echo "=========================================="
echo "Build VoxPopuli ASR: ${lang} (task ${SLURM_ARRAY_TASK_ID})"
echo "=========================================="

python -m audio_tokenization.utils.build_interleaved.shift_by_one \
    --parquet-dir "${CACHE_BASE}/${lang}" \
    --output-dir "${OUT_BASE}/${lang}" \
    --tokenizer-path "${TOKENIZER}" \
    --max-seq-len 262144 \
    --seq-threshold 8192 \
    --transcribe-ratio 0.4 \
    --tmp-dir /tmp \
    --num-workers 256

echo "Done: ${lang}"
