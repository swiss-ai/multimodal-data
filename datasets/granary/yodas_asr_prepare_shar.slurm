#!/bin/bash
# Granary YODAS → Shar (20TB, 92 language shards, NVIDIA Whisper pseudo-labels)
#SBATCH --account=infra01
#SBATCH --job-name=granary-shar
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/granary_shar_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/granary_shar_%A_%a.err
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
SHAR_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/granary_yodas

RANK="${SLURM_ARRAY_TASK_ID}"
WORLD="${SLURM_ARRAY_TASK_COUNT}"

# Map array index to multiple shards (92 shards across WORLD tasks)
mapfile -t ALL_SHARDS < <(ls "${PARQUET_ROOT}" | sort)
TOTAL=${#ALL_SHARDS[@]}

NODE_TAG="$(printf "%02d" "${RANK}")"
NODE_SHAR_DIR="${SHAR_ROOT}/node_${NODE_TAG}"

echo "=========================================="
echo "Granary YODAS → Shar"
echo "  Rank ${RANK}/${WORLD}, ${TOTAL} total shards, 32 workers"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname -s)"
echo "Output: ${NODE_SHAR_DIR}"
echo "=========================================="

mkdir -p ${REPO}/logs "${NODE_SHAR_DIR}"

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:${REPO}:${PYTHONPATH:-}
export PATH=/users/xyixuan/.local/ffmpeg/bin:$PATH
export LD_LIBRARY_PATH=/users/xyixuan/.local/ffmpeg/lib:${LD_LIBRARY_PATH:-}

ulimit -n 65536
uv pip install --python /opt/venv/bin/python polars

# Process shards assigned to this rank (round-robin)
for (( i=RANK; i<TOTAL; i+=WORLD )); do
    SHARD="${ALL_SHARDS[$i]}"
    echo "--- Processing shard ${SHARD} (${i}/${TOTAL}) ---"

    python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
        --parquet-dir "${PARQUET_ROOT}/${SHARD}/asr_only" \
        --parquet-glob '*.parquet' \
        --shar-dir "${NODE_SHAR_DIR}/${SHARD}" \
        --id-column utt_id \
        --audio-column audio \
        --text-column text \
        --duration-column duration \
        --language-column lang \
        --custom-columns original_audio_id original_audio_offset \
        --text-tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json \
        --target-sr 24000 \
        --shard-size 20000 \
        --num-workers 32
done

echo ""
echo "=========================================="
echo "Rank ${RANK} completed!"
echo "=========================================="

# Last task submits aggregate job
if [ "${RANK}" -eq "$((WORLD - 1))" ]; then
    sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} \
        --account=infra01 --partition=normal --reservation=PA-2338-RL \
        --environment=nemo_25_11 \
        --time=00:10:00 --nodes=1 --cpus-per-task=1 \
        --job-name=granary-agg \
        --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/granary_aggregate_%j.out \
        --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/granary_aggregate_%j.err \
        --wrap="export PYTHONPATH=${REPO}:/iopsstor/scratch/cscs/xyixuan/dev/lhotse:\${PYTHONPATH:-}; \
                python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
                    --aggregate ${SHAR_ROOT}"
    echo "Aggregate job submitted"
fi
