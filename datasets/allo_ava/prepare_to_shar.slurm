#!/bin/bash
# Convert Allo-AVA to Lhotse SHAR (stage_2, interleave-ready).
# 1,336 long recordings → 300K segments via --trim_to_supervisions.
# 361h English speech with timestamped transcripts.
# Universal clip IDs: {wav_stem}@{clip_num:06d}
#
# Usage:
#   sbatch prepare_to_shar.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=ava-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/allo_ava_shar_%A.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/allo_ava_shar_%A.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
RUNTIME_ENV_SH="${REPO_DIR}/scripts/source_lhotse_runtime.sh"
TEXT_TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

cd "${REPO_DIR}"
source "${RUNTIME_ENV_SH}"

echo "=========================================="
echo "Allo-AVA → SHAR (stage_2, interleave-ready)"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
  --recipe allo_ava \
  --corpus_dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___avalab___Allo-AVA \
  --split train \
  --trim_to_supervisions \
  --text_tokenizer "${TEXT_TOKENIZER}" \
  --shar_output_dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/allo_ava \
  --target_sample_rate 24000 \
  --num_workers 64 \
  --shar_shard_size 5000

echo "Done"
