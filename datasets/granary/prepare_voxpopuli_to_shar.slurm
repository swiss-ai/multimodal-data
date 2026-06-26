#!/bin/bash
# Convert Granary VoxPopuli to SHAR (ASR + AST).
# ASR: 23 languages, original text → stage_2/voxpopuli_asr/{lang}
# AST: 22 languages, English translation → stage_2/voxpopuli_ast/{lang}
# Interleave-ready: universal IDs via --input-clip-id-parser voxpopuli
#
# Array tasks 0-22: ASR (23 languages)
# Array tasks 23-44: AST (22 languages, no English)
#
# Usage:
#   sbatch --array=0-44 prepare_voxpopuli_to_shar.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=voxpop-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/voxpopuli_shar_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/voxpopuli_shar_%A_%a.err
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

VOXPOPULI_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/granary/voxpopuli-granary
SHAR_BASE=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2

cd "${REPO_DIR}"
source "${RUNTIME_ENV_SH}"
uv pip install --python /opt/venv/bin/python orjson

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert

# Build language lists
mapfile -t ASR_LANGS < <(ls "${VOXPOPULI_ROOT}/asr/" | sort)
mapfile -t AST_LANGS < <(ls "${VOXPOPULI_ROOT}/ast/" | sed 's/.jsonl.gz//' | sort)
NUM_ASR=${#ASR_LANGS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}

if [ "${TASK_ID}" -lt "${NUM_ASR}" ]; then
  # ASR task
  lang="${ASR_LANGS[${TASK_ID}]}"
  SHAR_DIR="${SHAR_BASE}/voxpopuli_asr/${lang}"
  MODE="asr"

  if [ -f "${SHAR_DIR}/_SUCCESS" ]; then
    echo "[voxpopuli_asr/${lang}] done, skipping"
    exit 0
  fi

  echo "=========================================="
  echo "VoxPopuli ASR: ${lang} (task ${TASK_ID})"
  echo "=========================================="

  python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
    --wds-shards "${VOXPOPULI_ROOT}/asr/${lang}/*.tar" \
    --external-metadata "${VOXPOPULI_ROOT}/asr/${lang}/uttid_text.jsonl.gz" \
    --id-field uttid \
    --text-field text \
    --shar-dir "${SHAR_DIR}" \
    --target-sr 24000 \
    --shard-size 20000 \
    --shar-format flac \
    --text-tokenizer "${TEXT_TOKENIZER}" \
    --input-clip-id-parser voxpopuli \
    --language "${lang}" \
    --num-workers 128
else
  # AST task
  AST_IDX=$(( TASK_ID - NUM_ASR ))
  lang="${AST_LANGS[${AST_IDX}]}"
  SHAR_DIR="${SHAR_BASE}/voxpopuli_ast/${lang}"
  MODE="ast"

  if [ -f "${SHAR_DIR}/_SUCCESS" ]; then
    echo "[voxpopuli_ast/${lang}] done, skipping"
    exit 0
  fi

  if [ ! -d "${VOXPOPULI_ROOT}/asr/${lang}" ]; then
    echo "[voxpopuli_ast/${lang}] no audio tars, skipping"
    exit 0
  fi

  echo "=========================================="
  echo "VoxPopuli AST: ${lang} (task ${TASK_ID})"
  echo "=========================================="

  python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
    --wds-shards "${VOXPOPULI_ROOT}/asr/${lang}/*.tar" \
    --external-metadata "${VOXPOPULI_ROOT}/ast/${lang}.jsonl.gz" \
    --id-field uttid \
    --text-field text_en \
    --custom-fields text_orig \
    --shar-dir "${SHAR_DIR}" \
    --target-sr 24000 \
    --shard-size 20000 \
    --shar-format flac \
    --text-tokenizer "${TEXT_TOKENIZER}" \
    --input-clip-id-parser voxpopuli \
    --language "${lang}" \
    --num-workers 128
fi

echo "Done: voxpopuli_${MODE}/${lang}"
