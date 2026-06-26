#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=gs2-shar
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/gs2_shar_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/gs2_shar_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --array=0-2

# GigaSpeech2 → Lhotse Shar (CPU-only, tar-streaming via --external-metadata)
#
# Reads tar.gz archives directly without extraction (~63M inodes avoided).
# Each array task converts one language (th=0, id=1, vi=2).
#
# Usage:
#   sbatch prepare_to_shar.slurm                # all 3 languages
#   sbatch --array=0 prepare_to_shar.slurm      # Thai only
#   sbatch --array=1 prepare_to_shar.slurm      # Indonesian only
#   sbatch --array=2 prepare_to_shar.slurm      # Vietnamese only

set -euo pipefail

LANGS=(th id vi)
LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

DATA_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___speechcolab___gigaspeech2/data
SHAR_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/gigaspeech2
TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json
NUM_WORKERS=128

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer:${PYTHONPATH:-}

uv pip install --python /opt/venv/bin/python orjson

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs

echo "=========================================="
echo "GigaSpeech2 → Shar: ${LANG}"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}, Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Language: ${LANG}"
echo "Input: ${DATA_ROOT}/${LANG}/train/*.tar.gz"
echo "Metadata: ${DATA_ROOT}/${LANG}/train_refined.tsv"
echo "Output: ${SHAR_ROOT}/${LANG}_train"
echo "Workers: ${NUM_WORKERS}"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
    --wds-shards "${DATA_ROOT}/${LANG}/train/*.tar.gz" \
    --external-metadata "${DATA_ROOT}/${LANG}/train_refined.tsv" \
    --shar-dir "${SHAR_ROOT}/${LANG}_train" \
    --target-sr 24000 \
    --min-sr 16000 \
    --shard-size 5000 \
    --shar-format flac \
    --num-workers ${NUM_WORKERS} \
    --text-tokenizer "${TOKENIZER}"

echo ""
echo "=========================================="
echo "${LANG} done!"
echo "=========================================="
