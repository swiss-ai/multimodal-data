#!/bin/bash
# Re-prep People's Speech Clean → Shar with normalized text + universal IDs
#SBATCH --account=infra01
#SBATCH --job-name=ps-shar-v2
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/ps-shar-v2_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/ps-shar-v2_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --reservation=PA-2338-RL
#SBATCH --exclusive
#SBATCH --container-writable

set -euo pipefail

REPO=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
ARROW_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/peoples_speech_cache/peoples_speech_final
SHAR_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/peoples_speech_clean_train_v2
METADATA=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/peoples_speech_clean_metadata/peoples_speech_clean.jsonl.zst
TEXT_TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

echo "=========================================="
echo "People's Speech Clean → Shar v2"
echo "  Node: $(hostname -s)"
echo "=========================================="

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
source "${REPO}/scripts/source_lhotse_runtime.sh"
uv pip install --python /opt/venv/bin/python zstandard

mapfile -t ALL_ARROWS < <(find "${ARROW_DIR}" -maxdepth 1 -type f -name '*.arrow' | sort)
echo "Found ${#ALL_ARROWS[@]} arrow files"

mkdir -p "${SHAR_ROOT}"

python -m audio_tokenization.utils.prepare_data.prepare_hf_to_shar \
    --arrow-files "${ALL_ARROWS[@]}" \
    --shar-dir "${SHAR_ROOT}" \
    --id-column id \
    --audio-column audio \
    --text-column text \
    --external-metadata "${METADATA}" \
    --text-field text_norm \
    --input-clip-id-parser trailing_number \
    --text-tokenizer "${TEXT_TOKENIZER}" \
    --target-sr 24000 \
    --shard-size 20000 \
    --shar-format flac \
    --num-workers 128 \
    --resampling-backend soxr

echo "Finished!"
