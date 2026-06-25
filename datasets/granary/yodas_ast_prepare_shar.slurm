#!/bin/bash
# Granary YODAS AST split → Shar (both transcribe + translate)
#SBATCH --account=infra01
#SBATCH --job-name=granary-ast
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/granary_ast_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/granary_ast_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --array=0-20
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --reservation=PA-2338-RL
#SBATCH --exclusive

set -euo pipefail

REPO=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
PARQUET_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/granary/yodas-granary/data
SHAR_TRANSCRIBE=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/granary_yodas_ast
SHAR_TRANSLATE=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/translate/granary_yodas_ast
TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

RANK="${SLURM_ARRAY_TASK_ID}"
WORLD="${SLURM_ARRAY_TASK_COUNT}"

mapfile -t ALL_SHARDS < <(
    for d in ${PARQUET_ROOT}/*/ast; do
        [ -d "$d" ] && basename "$(dirname "$d")"
    done | sort
)
TOTAL=${#ALL_SHARDS[@]}

NODE_TAG="$(printf "%02d" "${RANK}")"

echo "=========================================="
echo "Granary YODAS AST → Shar (transcribe + translate)"
echo "  Rank ${RANK}/${WORLD}, ${TOTAL} total shards, 32 workers"
echo "=========================================="

mkdir -p ${REPO}/logs

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:${REPO}:${PYTHONPATH:-}
export PATH=/users/xyixuan/.local/ffmpeg/bin:$PATH
export LD_LIBRARY_PATH=/users/xyixuan/.local/ffmpeg/lib:${LD_LIBRARY_PATH:-}

ulimit -n 65536
uv pip install --python /opt/venv/bin/python polars

for (( i=RANK; i<TOTAL; i+=WORLD )); do
    SHARD="${ALL_SHARDS[$i]}"
    PARQUET_DIR="${PARQUET_ROOT}/${SHARD}/ast"

    # Pass 1: transcribe (original text)
    echo "--- ${SHARD} transcribe ---"
    python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
        --parquet-dir "${PARQUET_DIR}" \
        --parquet-glob '*.parquet' \
        --shar-dir "${SHAR_TRANSCRIBE}/node_${NODE_TAG}/${SHARD}" \
        --id-column utt_id \
        --audio-column audio \
        --text-column text \
        --duration-column duration \
        --language-column lang \
        --custom-columns original_audio_id original_audio_offset \
        --text-tokenizer "${TOKENIZER}" \
        --target-sr 24000 \
        --shard-size 20000 \
        --num-workers 32

    # Pass 2: translate (English translation)
    echo "--- ${SHARD} translate ---"
    python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
        --parquet-dir "${PARQUET_DIR}" \
        --parquet-glob '*.parquet' \
        --shar-dir "${SHAR_TRANSLATE}/node_${NODE_TAG}/${SHARD}" \
        --id-column utt_id \
        --audio-column audio \
        --text-column translation_en \
        --duration-column duration \
        --language-column lang \
        --custom-columns original_audio_id original_audio_offset \
        --text-tokenizer "${TOKENIZER}" \
        --target-sr 24000 \
        --shard-size 20000 \
        --num-workers 32
done

echo ""
echo "=========================================="
echo "Rank ${RANK} completed!"
echo "=========================================="
