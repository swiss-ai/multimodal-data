#!/bin/bash
# Normalize People's Speech Clean text via LLM ITN
# 5025 arrow files, ~1.5M entries, 4 GPUs per node via bash parallelism
#SBATCH --account=infra01
#SBATCH --job-name=ps-norm
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/ps-norm-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/ps-norm-%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL
#SBATCH --container-writable
#SBATCH --exclusive

set -euo pipefail

REPO=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
INPUT_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/peoples_speech_cache/peoples_speech_final
OUTPUT_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/peoples_speech_clean_metadata
MODEL=/capstor/store/cscs/swissai/infra01/MLLM/model_baseline/Qwen3-8B-Instruct

RANK="${SLURM_ARRAY_TASK_ID}"
WORLD="${SLURM_ARRAY_TASK_COUNT}"

echo "=========================================="
echo "People's Speech text normalization (rank ${RANK}/${WORLD})"
echo "  Node: $(hostname -s)"
echo "=========================================="

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export NUMEXPR_MAX_THREADS=16

GPUS_PER_NODE=4
GLOBAL_WORLD_SIZE=$(( WORLD * GPUS_PER_NODE ))

mkdir -p "${OUTPUT_DIR}"

pids=()
for gpu in 0 1 2 3; do
    global_rank=$(( RANK * GPUS_PER_NODE + gpu ))
    cache_root="/tmp/vllm_cache_ps_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${gpu}"
    log_prefix="[node-rank ${RANK}/${WORLD} gpu ${gpu} global-rank ${global_rank}/${GLOBAL_WORLD_SIZE}]"

    echo "${log_prefix} launching"

    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export VLLM_CACHE_ROOT="${cache_root}"

        python -m audio_tokenization.utils.prepare_data.preprocess.normalizer.normalize_text \
            --input-dir "${INPUT_DIR}" \
            --input-format arrow \
            --id-column id \
            --text-column text \
            --model "${MODEL}" \
            --batch-size 4096 \
            --output "${OUTPUT_DIR}/peoples_speech_clean.jsonl.zst" \
            --rank "${global_rank}" \
            --world-size "${GLOBAL_WORLD_SIZE}"
    ) &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if [ "${status}" -ne 0 ]; then
    echo "One or more GPU workers failed" >&2
    exit "${status}"
fi

echo "Rank ${RANK} completed!"
