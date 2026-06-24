#!/bin/bash
# Set your SLURM account: #SBATCH --account=<your-account>
#SBATCH --job-name=transcribe
#SBATCH --output=logs/transcribe_%A_%a.out
#SBATCH --error=logs/transcribe_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=normal
#SBATCH --gpus-per-task=4
#SBATCH --array=0-7
#SBATCH --environment=ctranslate2-nemo-cudnn
# Uncomment and set if you have a SLURM reservation:
# #SBATCH --reservation=<your-reservation>

################################################################################
# Audio Transcription Job
################################################################################
# Batch-transcribes audio shards with faster-whisper across multiple GPUs.
# Each array task owns one node; GPUS_PER_NODE workers run in parallel,
# one per GPU. Workers divide the shard list by index modulo total workers.
#
# USAGE:
#   INPUT_DIR=/path/to/shards OUTPUT_DIR=/path/to/out sbatch job.slurm
#
# PARAMETERS (environment variables):
#   INPUT_DIR        - Directory containing input shards (required)
#   OUTPUT_DIR       - Directory for transcribed output (required)
#   INPUT_FORMAT     - Loader format: "arrow" (default: arrow)
#   LANGUAGE         - Whisper language code (default: en)
#   WHISPER_MODEL    - faster-whisper model name (default: turbo)
#   BATCH_SIZE       - Inference batch size per worker (default: 16)
#   GPUS_PER_NODE    - GPUs / workers per node (default: 4)
#   CLUSTER_REPO_HOME - Path to this repo on the cluster (default: SLURM_SUBMIT_DIR)
#
# EXAMPLES:
#   INPUT_DIR=/data/audio OUTPUT_DIR=/data/out sbatch job.slurm
#   LANGUAGE=de WHISPER_MODEL=large-v3 INPUT_DIR=/data/audio OUTPUT_DIR=/data/out sbatch job.slurm
################################################################################

set -euo pipefail

# ========================================
# Configuration
# ========================================

CLUSTER_REPO_HOME="${CLUSTER_REPO_HOME:-$SLURM_SUBMIT_DIR}"
INPUT_DIR="${INPUT_DIR:?INPUT_DIR must be set (path to input shards)}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set (path for transcribed output)}"
INPUT_FORMAT="${INPUT_FORMAT:-arrow}"
LANGUAGE="${LANGUAGE:-en}"
WHISPER_MODEL="${WHISPER_MODEL:-turbo}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

# ========================================
# Display Configuration
# ========================================

mkdir -p logs "$OUTPUT_DIR"

echo "=========================================="
echo "Audio Transcription Job"
echo "=========================================="
echo "Repo root:      $CLUSTER_REPO_HOME"
echo "Job ID:         $SLURM_JOB_ID"
echo "Array task:     $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Node:           $SLURM_NODELIST"
echo "Start:          $(date)"
echo ""
echo "Configuration:"
echo "  Input dir:    $INPUT_DIR"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Format:       $INPUT_FORMAT"
echo "  Language:     $LANGUAGE"
echo "  Model:        $WHISPER_MODEL"
echo "  Batch size:   $BATCH_SIZE"
echo "  GPUs/node:    $GPUS_PER_NODE"
echo "=========================================="

# ========================================
# Launch one worker per GPU
# ========================================

NUM_TOTAL_WORKERS=$(( SLURM_ARRAY_TASK_COUNT * GPUS_PER_NODE ))
BASE_WORKER_IDX=$(( SLURM_ARRAY_TASK_ID * GPUS_PER_NODE ))

echo "Array task ${SLURM_ARRAY_TASK_ID}: launching ${GPUS_PER_NODE} workers (IDs ${BASE_WORKER_IDX}..$(( BASE_WORKER_IDX + GPUS_PER_NODE - 1 )))"

PIDS=()
for i in $(seq 0 $(( GPUS_PER_NODE - 1 ))); do
    GLOBAL_WORKER_ID=$(( BASE_WORKER_IDX + i ))

    # Per-worker HF dataset cache to avoid lock contention
    export HF_DATASETS_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}/worker_${GLOBAL_WORKER_ID}"
    mkdir -p "$HF_DATASETS_CACHE"

    echo "Worker ${GLOBAL_WORKER_ID} on GPU ${i}"

    CUDA_VISIBLE_DEVICES=$i python "${CLUSTER_REPO_HOME}/transcription/transcribe.py" \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --input_format "$INPUT_FORMAT" \
        --language "$LANGUAGE" \
        --whisper_model "$WHISPER_MODEL" \
        --batch_size "$BATCH_SIZE" \
        --worker_id "$GLOBAL_WORKER_ID" \
        --num_workers "$NUM_TOTAL_WORKERS" &

    PIDS+=($!)
done

# ========================================
# Wait and report
# ========================================

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "Worker PID $pid failed (exit $?)."
        FAILED=$(( FAILED + 1 ))
    fi
done

echo ""
echo "=========================================="
echo "Completed at: $(date)"
if [ "$FAILED" -gt 0 ]; then
    echo "${FAILED}/${GPUS_PER_NODE} workers failed on this node."
    exit 1
fi
echo "All ${GPUS_PER_NODE} workers completed successfully."
echo "=========================================="
