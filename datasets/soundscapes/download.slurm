#!/bin/bash
# Download laion/soundscapes from HuggingFace — one batch per array task
# ~22.5TB total, 6776 tar files across 100 batches (~225GB per batch)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-soundscapes
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/soundscapes-%x-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/soundscapes-%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL
#SBATCH --array=1-30

set -euo pipefail

DATASET_REPO="laion/soundscapes"
REVISION="97ac761b027e9bb7dd781f069b45cf1ec425624a"  # main as of 2026-03-28
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___laion___soundscapes"

# 100 batches across 30 nodes: ~3-4 batches per node
# Task 1: batch 1-4, Task 2: batch 5-7, ... Task 30: batch 98-100
TOTAL_BATCHES=100
TOTAL_TASKS=30
PER_TASK=$(( (TOTAL_BATCHES + TOTAL_TASKS - 1) / TOTAL_TASKS ))
BATCH_START=$(( (SLURM_ARRAY_TASK_ID - 1) * PER_TASK + 1 ))
BATCH_END=$(( BATCH_START + PER_TASK - 1 ))
if [ "$BATCH_END" -gt "$TOTAL_BATCHES" ]; then
    BATCH_END=$TOTAL_BATCHES
fi

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache_task_${SLURM_ARRAY_TASK_ID}"

echo "========================================"
echo "Soundscapes download: batch_$(printf '%04d' $BATCH_START)-batch_$(printf '%04d' $BATCH_END) (task ${SLURM_ARRAY_TASK_ID})"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

mkdir -p "${DEST_DIR}"

for b in $(seq "$BATCH_START" "$BATCH_END"); do
    BATCH=$(printf 'batch_%04d' "$b")
    echo "[$(date '+%F %T')] Downloading ${BATCH}..."
    huggingface-cli download "${DATASET_REPO}" \
      --repo-type dataset \
      --revision "${REVISION}" \
      --local-dir "${DEST_DIR}" \
      --include "${BATCH}_*.tar" \
      --max-workers 64
    echo "[$(date '+%F %T')] Done: ${BATCH}"
done

echo "[$(date '+%F %T')] All batches finished for task ${SLURM_ARRAY_TASK_ID}"
echo "Pinned to commit: ${REVISION}"

rm -rf "${HF_HUB_CACHE}"
echo "Done."
