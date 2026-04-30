#!/bin/bash
# Re-prep Zeroth Korean → Shar with universal IDs
#SBATCH --account=infra01
#SBATCH --job-name=zeroth-shar
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/zeroth-shar-v2_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/zeroth-shar-v2_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --reservation=PA-2338-RL
#SBATCH --exclusive
#SBATCH --container-writable

set -euo pipefail

REPO=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
PARQUET_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/kresnik___zeroth_korean/data
SHAR_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/zeroth_korean_train_v2
TEXT_TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

echo "=========================================="
echo "Zeroth Korean → Shar v2"
echo "  Node: $(hostname -s)"
echo "=========================================="

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
source "${REPO}/scripts/source_lhotse_runtime.sh"
uv pip install --python /opt/venv/bin/python polars

mkdir -p "${SHAR_ROOT}"

python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${PARQUET_DIR}" \
    --parquet-glob 'train-*.parquet' \
    --shar-dir "${SHAR_ROOT}" \
    --id-column id \
    --audio-column audio \
    --text-column text \
    --input-clip-id-parser trailing_number \
    --text-tokenizer "${TEXT_TOKENIZER}" \
    --target-sr 24000 \
    --shard-size 6000 \
    --num-workers 4 \
    --resampling-backend soxr

echo "Finished!"
