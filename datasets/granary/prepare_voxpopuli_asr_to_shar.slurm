#!/bin/bash
# Convert Granary VoxPopuli ASR to SHAR — one language per array task.
# 23 languages, original text transcription → stage_2/voxpopuli_asr/{lang}
# Interleave-ready: universal IDs via --input-clip-id-parser voxpopuli
#
# Usage:
#   sbatch --array=0-22 prepare_voxpopuli_asr_to_shar.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=voxpop-asr
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/granary_voxpopuli_asr_shar_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/granary_voxpopuli_asr_shar_%A_%a.err
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

# Map array task ID to language
LANGS=($(ls "${VOXPOPULI_ROOT}/asr/" | sort))
lang="${LANGS[${SLURM_ARRAY_TASK_ID}]}"

SHAR_DIR="${SHAR_BASE}/voxpopuli_asr/${lang}"
if [ -f "${SHAR_DIR}/_SUCCESS" ]; then
  echo "[voxpopuli_asr/${lang}] already done, skipping"
  exit 0
fi

echo "=========================================="
echo "VoxPopuli ASR → SHAR: lang=${lang}"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
  --wds-shards "${VOXPOPULI_ROOT}/asr/${lang}"/*.tar \
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
  --num-workers 64

echo "[voxpopuli_asr/${lang}] done"
