#!/usr/bin/env bash
# SHAR conversion examples (copy/paste templates).
# This script only prints commands; it does not run conversions.

set -euo pipefail

cat <<'EOF'
# ==========================================================
# Shared setup (adjust once)
# ==========================================================
PYTHON_BIN=/opt/venv/bin/python
BENCHMARK_REPO=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
TEXT_TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

export PYTHONPATH="${BENCHMARK_REPO}:${PYTHONPATH:-}"

# ==========================================================
# 1) HF Arrow -> SHAR (prepare_hf_to_shar)
# ==========================================================
# Example: Audiocite processed
"${PYTHON_BIN}" -m audio_tokenization.utils.prepare_data.prepare_hf_to_shar \
  --arrow-dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/audiocite/audiocite_processed \
  --arrow-glob 'data-*.arrow' \
  --shar-dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/audiocite_processed \
  --audio-column audio \
  --text-column text \
  --id-column id \
  --target-sr 24000 \
  --shard-size 5000 \
  --shar-format flac \
  --text-tokenizer "${TEXT_TOKENIZER}" \
  --num-workers 288 \
  --resampling-backend sox

# ==========================================================
# 2) Parquet -> SHAR (prepare_parquet_to_shar)
# ==========================================================
# Example: CoRal train
"${PYTHON_BIN}" -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
  --parquet-dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/coral/read_aloud \
  --parquet-glob 'train-*.parquet' \
  --shar-dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/coral_train \
  --id-column id_recording \
  --text-column text \
  --target-sr 24000 \
  --text-tokenizer "${TEXT_TOKENIZER}" \
  --shard-size 5000 \
  --num-workers 128 \
  --parquet-read-mode pyarrow-batched \
  --parquet-batch-size 128 \
  --ffmpeg-bin /capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64/bin/ffmpeg

# ==========================================================
# 3) Lhotse recipe -> SHAR (prepare_lhotse_recipe_to_shar)
# ==========================================================
# Example: CommonVoice es train
"${PYTHON_BIN}" -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
  --recipe commonvoice \
  --corpus_dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/commonvoice24/cv-corpus-24.0-2025-12-05 \
  --split train \
  --language es \
  --target_sample_rate 24000 \
  --text_tokenizer "${TEXT_TOKENIZER}" \
  --num_workers 288 \
  --shar_output_dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/commonvoice/es_train

# Example: CommonVoice es other (stage_1)
"${PYTHON_BIN}" -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
  --recipe commonvoice \
  --corpus_dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/commonvoice24/cv-corpus-24.0-2025-12-05 \
  --split other \
  --language es \
  --target_sample_rate 24000 \
  --text_tokenizer "${TEXT_TOKENIZER}" \
  --num_workers 288 \
  --shar_output_dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/commonvoice/es_other

# ==========================================================
# 4) WDS tar shards -> SHAR (prepare_wds_to_shar)
# ==========================================================
# Example: Suno shards_s1
"${PYTHON_BIN}" -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
  --wds-shards '/capstor/store/cscs/swissai/infra01/audio-datasets/raw/suno/shards_s1/*.tar' \
  --shar-dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/suno_s1 \
  --target-sr 24000 \
  --min-sr 16000 \
  --shard-size 5000 \
  --shar-format flac \
  --text-tokenizer "${TEXT_TOKENIZER}" \
  --num-workers 128

# ==========================================================
# 5) Audio-dir + JSONL(VAD) -> SHAR (prepare_audio_dir_to_shar)
# ==========================================================
# Example: VoxPopuli
"${PYTHON_BIN}" -m audio_tokenization.utils.prepare_data.prepare_audio_dir_to_shar \
  --audio-root /capstor/store/cscs/swissai/infra01/audio-datasets/voxpopuli/raw_audios \
  --jsonl-files /capstor/store/cscs/swissai/infra01/audio-datasets/voxpopuli/vad_results_merged/*.jsonl \
  --shar-dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/voxpopuli_train \
  --audio-ext .ogg \
  --target-sr 24000 \
  --shar-format flac \
  --shard-size 5000 \
  --num-workers 288 \
  --min-sr 16000
EOF
