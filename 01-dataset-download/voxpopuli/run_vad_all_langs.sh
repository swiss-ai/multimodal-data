#!/bin/bash
# Run Silero VAD on all VoxPopuli languages that don't have results yet.
# Existing results (en, de, es, it) are skipped automatically via resume.
#
# Usage (called from slurm script):
#   bash run_vad_all_langs.sh

set -euo pipefail

# Install dependencies into the nemo container
uv pip install --system --break-system-packages --no-deps --no-build-isolation \
    git+https://github.com/pytorch/audio.git@release/2.9
uv pip install --system --break-system-packages --no-deps \
    onnxruntime speechbrain hyperpyyaml silero-vad orjson

AUDIO_ROOT="/capstor/store/cscs/swissai/infra01/audio-datasets/voxpopuli/raw_audios"
OUTPUT_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/voxpopuli/vad_results"
NUM_WORKERS=288
DATASET="voxpopuli"

LANGS=(cs et fi fr hr hu nl pl ro sk sl)

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "VoxPopuli VAD — ${#LANGS[@]} languages"
echo "Workers: ${NUM_WORKERS}"
echo "=========================================="

for lang in "${LANGS[@]}"; do
    AUDIO_DIR="${AUDIO_ROOT}/${lang}"

    if [ ! -d "${AUDIO_DIR}" ]; then
        echo "[${lang}] Audio dir not found: ${AUDIO_DIR}, skipping"
        continue
    fi

    echo ""
    echo "[$(date '+%F %T')] Starting VAD for ${lang}..."

    python -m audio_tokenization.utils.prepare_data.run_vad \
        --audio_dir "${AUDIO_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --dataset "${DATASET}" \
        --num_workers "${NUM_WORKERS}" \
        --backend onnx \
        --pool_start fork

    echo "[$(date '+%F %T')] Done: ${lang}"
done

echo ""
echo "=========================================="
echo "All languages complete"
echo ""
for f in "${OUTPUT_DIR}"/*.jsonl; do
    count=$(wc -l < "$f")
    printf "  %-30s %'d entries\n" "$(basename "$f")" "${count}"
done
echo "=========================================="
