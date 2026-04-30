#!/bin/bash
# Convert Granary YTC to SHAR (ASR + AST).
# ASR: 23 languages, native text → stage_2/granary_ytc_asr/{lang}
# AST: 22 languages (no English), English translation → stage_2/granary_ytc_ast/{lang}
# Audio tars: ytc-granary/asr/{lang}/*.tar
# NVIDIA metadata: nvidia-granary/{lang}/ytc/{lang}_asr.jsonl / {lang}_ast-en.jsonl
# Interleave-ready: clip IDs parsed by --input-clip-id-parser ytc
#
# Array tasks 0-22: ASR (23 languages)
# Array tasks 23-44: AST (22 languages, no English)
#
# Usage:
#   sbatch --array=0-44 prepare_ytc_to_shar.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=ytc-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/ytc_shar_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/ytc_shar_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
RUNTIME_ENV_SH="${REPO_DIR}/scripts/utils/source_lhotse_runtime.sh"
TEXT_TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

YTC_AUDIO=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/granary/ytc-granary/asr
NVIDIA_META=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/granary/nvidia-granary
SHAR_BASE=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2

cd "${REPO_DIR}"
source "${RUNTIME_ENV_SH}"
uv pip install --python /opt/venv/bin/python orjson

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert

# ASR languages (sorted)
ASR_LANGS=(bg cs da de el en es et fi fr hr hu it lt lv nl pl pt ro sk sl sv uk)
# AST languages (no English)
AST_LANGS=(bg cs da de el es et fi fr hr hu it lt lv nl pl pt ro sk sl sv uk)
NUM_ASR=${#ASR_LANGS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}

if [ "${TASK_ID}" -lt "${NUM_ASR}" ]; then
  # ASR task
  lang="${ASR_LANGS[${TASK_ID}]}"
  SHAR_DIR="${SHAR_BASE}/granary_ytc_asr/${lang}"
  META_FILE="${NVIDIA_META}/${lang}/ytc/${lang}_asr.jsonl"

  if [ -f "${SHAR_DIR}/_SUCCESS" ]; then
    echo "[granary_ytc_asr/${lang}] done, skipping"
    exit 0
  fi

  echo "=========================================="
  echo "YTC ASR: ${lang} (task ${TASK_ID})"
  echo "=========================================="

  python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
    --wds-shards "${YTC_AUDIO}/${lang}/*.tar" \
    --external-metadata "${META_FILE}" \
    --id-field utt_id \
    --text-field text \
    --custom-fields source_lang dataset_source offset \
    --shar-dir "${SHAR_DIR}" \
    --target-sr 24000 \
    --shard-size 20000 \
    --text-tokenizer "${TEXT_TOKENIZER}" \
    --input-clip-id-parser ytc \
    --language "${lang}" \
    --num-workers 128
else
  # AST task
  AST_IDX=$(( TASK_ID - NUM_ASR ))
  lang="${AST_LANGS[${AST_IDX}]}"
  SHAR_DIR="${SHAR_BASE}/granary_ytc_ast/${lang}"
  META_FILE="${NVIDIA_META}/${lang}/ytc/${lang}_ast-en.jsonl"

  if [ -f "${SHAR_DIR}/_SUCCESS" ]; then
    echo "[granary_ytc_ast/${lang}] done, skipping"
    exit 0
  fi

  echo "=========================================="
  echo "YTC AST: ${lang} (task ${TASK_ID})"
  echo "=========================================="

  python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
    --wds-shards "${YTC_AUDIO}/${lang}/*.tar" \
    --external-metadata "${META_FILE}" \
    --id-field utt_id \
    --text-field answer \
    --custom-fields source_lang text \
    --shar-dir "${SHAR_DIR}" \
    --target-sr 24000 \
    --shard-size 20000 \
    --text-tokenizer "${TEXT_TOKENIZER}" \
    --input-clip-id-parser ytc \
    --language "${lang}" \
    --num-workers 128
fi

echo "Done: ${lang}"
