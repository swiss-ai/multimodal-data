#!/bin/bash
# Clean GigaSpeech v1 text + LLM ITN
# 258 parquets, 4 GPUs per node, ~16 parquets per node
# Launch 4 independent offline workers per node, one per GPU.
#SBATCH --account=infra01
#SBATCH --job-name=gs-clean
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/gs-clean-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/gs-clean-%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL
#SBATCH --container-writable
#SBATCH --exclusive

set -euo pipefail

REPO=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
PARQUET_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___speechcolab___gigaspeech/parquet-data/xl
OUTPUT_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___speechcolab___gigaspeech/metadata
MODEL=/capstor/store/cscs/swissai/infra01/MLLM/model_baseline/Qwen3-8B-Instruct

RANK="${SLURM_ARRAY_TASK_ID}"
WORLD="${SLURM_ARRAY_TASK_COUNT}"

echo "=========================================="
echo "GigaSpeech text cleaning (rank ${RANK}/${WORLD})"
echo "  Node: $(hostname -s)"
echo "=========================================="

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export NUMEXPR_MAX_THREADS=16

GPUS_PER_NODE=4
GLOBAL_WORLD_SIZE=$(( WORLD * GPUS_PER_NODE ))

pids=()
for gpu in 0 1 2 3; do
    global_rank=$(( RANK * GPUS_PER_NODE + gpu ))
    cache_root="/tmp/vllm_cache_gs_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${gpu}"
    log_prefix="[node-rank ${RANK}/${WORLD} gpu ${gpu} global-rank ${global_rank}/${GLOBAL_WORLD_SIZE}]"

    echo "${log_prefix} launching"

    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export VLLM_CACHE_ROOT="${cache_root}"

        python -m audio_tokenization.utils.prepare_data.preprocess.clean_gigaspeech \
            --parquet-dir "${PARQUET_DIR}" \
            --output-dir "${OUTPUT_DIR}" \
            --model "${MODEL}" \
            --batch-size 4096 \
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
