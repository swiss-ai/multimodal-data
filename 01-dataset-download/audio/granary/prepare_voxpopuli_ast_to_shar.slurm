#!/bin/bash
# Convert Granary VoxPopuli AST to SHAR — one language per array task.
# 22 languages, English translation → stage_2/voxpopuli_ast/{lang}
# Uses same audio tars as ASR, different text metadata.
# Interleave-ready: universal IDs via --input-clip-id-parser voxpopuli
#
# Usage:
#   sbatch --array=0-21 prepare_voxpopuli_ast_to_shar.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=voxpop-ast
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/granary_voxpopuli_ast_shar_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/granary_voxpopuli_ast_shar_%A_%a.err
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

VOXPOPULI_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/granary/voxpopuli-granary
SHAR_BASE=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2

cd "${REPO_DIR}"
source "${RUNTIME_ENV_SH}"
uv pip install --python /opt/venv/bin/python orjson

# Map array task ID to language (from ast/*.jsonl.gz filenames)
LANGS=($(ls "${VOXPOPULI_ROOT}/ast/"*.jsonl.gz | xargs -I{} basename {} .jsonl.gz | sort))
lang="${LANGS[${SLURM_ARRAY_TASK_ID}]}"

SHAR_DIR="${SHAR_BASE}/voxpopuli_ast/${lang}"
if [ -f "${SHAR_DIR}/_SUCCESS" ]; then
  echo "[voxpopuli_ast/${lang}] already done, skipping"
  exit 0
fi

if [ ! -d "${VOXPOPULI_ROOT}/asr/${lang}" ]; then
  echo "[voxpopuli_ast/${lang}] no audio tars, skipping"
  exit 0
fi

echo "=========================================="
echo "VoxPopuli AST → SHAR: lang=${lang}"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
  --wds-shards "${VOXPOPULI_ROOT}/asr/${lang}"/*.tar \
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
  --num-workers 64

echo "[voxpopuli_ast/${lang}] done"
