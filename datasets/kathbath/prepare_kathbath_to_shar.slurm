#!/bin/bash
# Reconvert Kathbath to SHAR with language metadata.
# 12 Indic languages, ~1460h, read-aloud speech.
# One array task per language.
#
# Usage:
#   sbatch --array=0-11 prepare_kathbath_to_shar.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=kathbath-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/kathbath_shar_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/kathbath_shar_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
RUNTIME_ENV_SH="${REPO_DIR}/scripts/utils/source_lhotse_runtime.sh"
TEXT_TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

RAW_BASE=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___ai4bharat___Kathbath
SHAR_BASE=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/kathbath_v2

cd "${REPO_DIR}"
source "${RUNTIME_ENV_SH}"
uv pip install --python /opt/venv/bin/python orjson

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert

LANGS=(bengali gujarati hindi kannada malayalam marathi odia punjabi sanskrit tamil telugu urdu)
lang="${LANGS[${SLURM_ARRAY_TASK_ID}]}"
SHAR_DIR="${SHAR_BASE}/${lang}"

if [ -f "${SHAR_DIR}/_SUCCESS" ]; then
    echo "[kathbath/${lang}] done, skipping"
    exit 0
fi

echo "=========================================="
echo "Kathbath ${lang} → SHAR (task ${SLURM_ARRAY_TASK_ID})"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${RAW_BASE}/${lang}" \
    --parquet-glob "train-*.parquet" \
    --shar-dir "${SHAR_DIR}" \
    --id-column fname \
    --text-column text \
    --audio-column audio_filepath \
    --language-column lang \
    --custom-columns gender speaker_id \
    --text-tokenizer "${TEXT_TOKENIZER}" \
    --target-sr 24000 \
    --shard-size 20000 \
    --num-workers 128

echo "Done: kathbath/${lang}"
