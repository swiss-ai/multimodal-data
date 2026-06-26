#!/bin/bash
# Reconvert Legco Speech to SHAR with full metadata.
# ~21.8Kh, Cantonese parliamentary speech.
# Includes: video_id, segment_id, start_time, end_time, language in custom.
# Interleave-ready: source_id from trailing_number parser (video_id_segmentId).
#
# Usage:
#   sbatch prepare_legco_to_shar.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=legco-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/legco_shar_%A.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/legco_shar_%A.err
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

RAW_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/legco-speech/segmented
SHAR_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/legco_speech_train_v2

cd "${REPO_DIR}"
source "${RUNTIME_ENV_SH}"
uv pip install --python /opt/venv/bin/python orjson

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert

if [ -f "${SHAR_DIR}/_SUCCESS" ]; then
    echo "[legco_speech] done, skipping"
    exit 0
fi

echo "=========================================="
echo "Legco Speech → SHAR"
echo "Raw: ${RAW_DIR}"
echo "Output: ${SHAR_DIR}"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${RAW_DIR}" \
    --parquet-glob "*.parquet" \
    --shar-dir "${SHAR_DIR}" \
    --id-column video_id \
    --text-column text \
    --audio-column audio \
    --custom-columns video_id segment_id start_time end_time duration \
    --text-tokenizer "${TEXT_TOKENIZER}" \
    --target-sr 24000 \
    --shard-size 20000 \
    --language zh-hk \
    --num-workers 128

echo "Done: legco_speech"
