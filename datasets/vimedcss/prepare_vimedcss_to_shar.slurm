#!/bin/bash
# Convert ViMedCSS to SHAR with full metadata.
# ~23h, Vietnamese medical conversational speech.
#
# Usage:
#   sbatch prepare_vimedcss_to_shar.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=vimedcss-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/vimedcss_shar_%A.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/vimedcss_shar_%A.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
RUNTIME_ENV_SH="${REPO_DIR}/scripts/utils/source_lhotse_runtime.sh"
TEXT_TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

RAW_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/tensorxt___ViMedCSS/data
SHAR_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/vimedcss_train

cd "${REPO_DIR}"
source "${RUNTIME_ENV_SH}"
uv pip install --python /opt/venv/bin/python orjson

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert

echo "=========================================="
echo "ViMedCSS → SHAR"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${RAW_DIR}" \
    --parquet-glob "train-*.parquet" \
    --shar-dir "${SHAR_DIR}" \
    --id-column segment_id \
    --text-column segment_text \
    --audio-column audio \
    --custom-columns start_time end_time duration_seconds topic cs_terms_count \
    --text-tokenizer "${TEXT_TOKENIZER}" \
    --target-sr 24000 \
    --shard-size 20000 \
    --language vi \
    --num-workers 30

echo "Done"
