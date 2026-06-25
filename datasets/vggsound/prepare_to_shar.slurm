#!/bin/bash
# Convert VGGSound train to SHAR (SHAR_TODO/annotate, pending captioning).
# 184K 10s clips, 310 event categories, stereo 16kHz.
# Event labels stored as supervision.text + custom.label.
#
# Usage:
#   sbatch prepare_to_shar.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=vgg-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/vggsound_shar_%A.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/vggsound_shar_%A.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
RUNTIME_ENV_SH="${REPO_DIR}/scripts/source_lhotse_runtime.sh"

cd "${REPO_DIR}"
source "${RUNTIME_ENV_SH}"

echo "=========================================="
echo "VGGSound → SHAR (SHAR_TODO/annotate)"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
  --recipe vggsound \
  --corpus_dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/javisdata-audios/audio/VGGSound \
  --split train \
  --shar_output_dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR_TODO/annotate/vggsound \
  --target_sample_rate 24000 \
  --num_workers 64 \
  --shar_shard_size 20000

echo "Done"
