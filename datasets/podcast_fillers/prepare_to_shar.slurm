#!/bin/bash
# Convert podcast-fillers to Lhotse SHAR (stage_2, interleave-ready).
# 17,942 clips from 100 podcasts, English, with transcription.
# Clips ordered by source_podcast + start time for interleaving.
#
# Usage:
#   sbatch prepare_to_shar.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=podcast-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/podcast_fillers_shar_%A.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/podcast_fillers_shar_%A.err
#SBATCH --time=01:00:00
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
DATA_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___ylacombe___podcast_fillers_processed

export PYTHONPATH="${LHOTSE_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
export PATH="${FFMPEG_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}"
uv pip install --python /opt/venv/bin/python --no-deps --force-reinstall "${TORCHCODEC_WHL}"
uv pip install --python /opt/venv/bin/python polars

echo "=========================================="
echo "podcast-fillers → SHAR (stage_2)"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
  --parquet-dir "${DATA_ROOT}" \
  --parquet-glob "train_with_clip_id.parquet" \
  --shar-dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/podcast_fillers \
  --id-column clip_id \
  --text-column text \
  --audio-column audio \
  --custom-columns source_podcast speaker start end dnsmos clip_num \
  --text-tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json \
  --target-sr 24000 \
  --shard-size 5000 \
  --num-workers 16

echo "Done"
