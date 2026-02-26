#!/bin/bash
#SBATCH --job-name=transcribe_data
#SBATCH --account=infra01
#SBATCH --output=logs/whisper_ps_%A_%a.out
#SBATCH --error=errs/whisper_ps_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=normal
#SBATCH --gpus=4
#SBATCH --array=0-7
#SBATCH --environment=ctranslate2-nemo-cudnn
#SBATCH --reservation=PA-2338-RL

# ----------------------------- Workflow -----------------------------------
# For people speech
# Each array gets 1 node. 4 workers are created assigning (4 CPU, 1 GPU 32GB).
# ---------------------------------------------------------------------------

USERNAME=arsaikia
INPUT_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/peoples_speech_cache/MLCommons___peoples_speech/dirty/0.0.0/f10597c5d3d3a63f8b6827701297c3afdf178272"
OUTPUT_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/peoples_speech_cache/updated_transcription"
WORK_DIR="/capstor/scratch/cscs/${USERNAME}/people_speech_transcribe"
INPUT_FORMAT="arrow"   
LANGUAGE="en"
GPUS_PER_NODE=4
export HF_HOME="/capstor/scratch/cscs/${USERNAME}/hf_cache"
export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib.real:${LD_LIBRARY_PATH}"

cd ${WORK_DIR}

NUM_TOTAL_WORKERS=$(( 8 * GPUS_PER_NODE ))  # array tasks × GPUs
BASE_WORKER_IDX=$(( SLURM_ARRAY_TASK_ID * GPUS_PER_NODE ))

echo "Array Task ${SLURM_ARRAY_TASK_ID}: launching ${GPUS_PER_NODE} workers (IDs ${BASE_WORKER_IDX}..$(( BASE_WORKER_IDX + GPUS_PER_NODE - 1 )))"

# Launch one worker per GPU
PIDS=()
for i in $(seq 0 $(( GPUS_PER_NODE - 1 ))); do
    GLOBAL_WORKER_ID=$(( BASE_WORKER_IDX + i ))

    # Per-worker HF dataset cache to avoid lock contention
    export HF_DATASETS_CACHE="${HF_HOME}/worker_${GLOBAL_WORKER_ID}"
    mkdir -p "$HF_DATASETS_CACHE"

    echo "Worker ${GLOBAL_WORKER_ID} on GPU ${i}"

    CUDA_VISIBLE_DEVICES=$i python transcribe.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --input_format "$INPUT_FORMAT" \
        --language "$LANGUAGE" \
        --worker_id "$GLOBAL_WORKER_ID" \
        --num_workers "$NUM_TOTAL_WORKERS" &

    PIDS+=($!)
done

# Report failed cases
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "Worker PID $pid failed (exit $?)."
        FAILED=$(( FAILED + 1 ))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "${FAILED}/${GPUS_PER_NODE} workers failed on this node."
    exit 1
fi

echo "All ${GPUS_PER_NODE} workers completed successfully."