#!/bin/bash
# Convert all MRSAudio splits to Lhotse SHAR.
# Speech/dialogue: lhotse recipe → stage_2 (with text supervision)
# Music/sing: standalone merged script → SHAR_TODO/annotate (audio-only)
#
# Usage:
#   sbatch prepare_to_shar.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=mrs-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/mrsaudio_shar_%A.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/mrsaudio_shar_%A.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
LHOTSE_DIR=/iopsstor/scratch/cscs/xyixuan/dev/lhotse
CORPUS_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___MRSAudio___MRSAudio/hf___MRSAudio___MRSAudio_git_clone
SHAR_BASE=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2
TEXT_TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

export PYTHONPATH="${LHOTSE_DIR}:${REPO_DIR}:${PYTHONPATH:-}"

echo "=========================================="
echo "MRSAudio → SHAR (speech + dialogue)"
echo "=========================================="

# Speech: 122K segments
python -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
  --recipe mrsaudio \
  --corpus_dir "${CORPUS_DIR}" \
  --split speech \
  --shar_output_dir "${SHAR_BASE}/mrsaudio_speech" \
  --target_sample_rate 24000 \
  --num_workers 64 \
  --shar_shard_size 5000 \
  --text_tokenizer "${TEXT_TOKENIZER}"

echo "Speech done"

# Dialogue: 70K segments
python -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
  --recipe mrsaudio \
  --corpus_dir "${CORPUS_DIR}" \
  --split dialogue \
  --shar_output_dir "${SHAR_BASE}/mrsaudio_dialogue" \
  --target_sample_rate 24000 \
  --num_workers 64 \
  --shar_shard_size 5000 \
  --text_tokenizer "${TEXT_TOKENIZER}"

echo "Dialogue done"

SCRIPT_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/mrsaudio
SHAR_ANNOTATE=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR_TODO/annotate

# Music: 36K segments → ~10K merged ≤30s chunks (audio-only)
python "${SCRIPT_DIR}/prepare_music_sing_to_shar.py" \
  --data-root "${CORPUS_DIR}" \
  --split MRSMusic \
  --shar-dir "${SHAR_ANNOTATE}/mrsaudio_music" \
  --num-workers 32

echo "Music done"

# Sing: 18K segments → ~6K merged ≤30s chunks (audio-only)
python "${SCRIPT_DIR}/prepare_music_sing_to_shar.py" \
  --data-root "${CORPUS_DIR}" \
  --split MRSSing \
  --shar-dir "${SHAR_ANNOTATE}/mrsaudio_sing" \
  --num-workers 32

echo "Sing done"
