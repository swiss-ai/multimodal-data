#!/bin/bash
# Download alakxender/dhivehi-audios-82-spk (~446GB, Dhivehi/Maldivian, MIT)
# 45 speaker dirs (f_00-f_14, m_00-m_29). Split across 9 array tasks.
#
# Usage:
#   sbatch --array=0-8 download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-dhivehi
#SBATCH --reservation=SD-69241-apertus-1-5-6
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/dhivehi-%x-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/dhivehi-%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

DATASET_REPO="alakxender/dhivehi-audios-82-spk"
REVISION="f2c2eeb9a7bde123455fbb0d1236e97b7964ac05"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___alakxender___dhivehi-audios-82-spk"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache_${SLURM_ARRAY_TASK_ID}"

uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# 45 speaker dirs split into 9 tasks of 5 each
ALL_DIRS=(
    f_00 f_01 f_02 f_03 f_04
    f_05 f_06 f_07 f_08 f_09
    f_10 f_11 f_12 f_13 f_14
    m_00 m_01 m_02 m_03 m_04
    m_05 m_06 m_07 m_08 m_09
    m_10 m_11 m_12 m_13 m_14
    m_15 m_16 m_17 m_18 m_19
    m_20 m_21 m_22 m_23 m_24
    m_25 m_26 m_27 m_28 m_29
)

PER_TASK=5
START=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
END=$(( START + PER_TASK ))
if [ $END -gt ${#ALL_DIRS[@]} ]; then END=${#ALL_DIRS[@]}; fi

INCLUDE_ARGS=()
for (( j=START; j<END; j++ )); do
    INCLUDE_ARGS+=( --include "${ALL_DIRS[$j]}/*" )
done

mkdir -p "${DEST_DIR}"

echo "========================================"
echo "Dhivehi download (task ${SLURM_ARRAY_TASK_ID})"
echo "Speakers: ${ALL_DIRS[@]:$START:$PER_TASK}"
echo "Dest: ${DEST_DIR}"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    "${INCLUDE_ARGS[@]}" \
    --local-dir "${DEST_DIR}" \
    --max-workers 64

# Task 0 also fetches README
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    huggingface-cli download "${DATASET_REPO}" \
        --repo-type dataset \
        --revision "${REVISION}" \
        --include "README.md" \
        --local-dir "${DEST_DIR}" \
        --max-workers 64
fi

rm -rf "${HF_HUB_CACHE}"
echo "[$(date '+%F %T')] Done: task ${SLURM_ARRAY_TASK_ID}"
