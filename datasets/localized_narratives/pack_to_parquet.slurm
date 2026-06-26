#!/bin/bash
# Pack Localized Narratives raw tree → per-group parquet (one node per group).
#
# Storage layout:
#   in:  /capstor/store/cscs/swissai/infra01/audio-datasets/raw/localized_narratives
#   out: /capstor/store/cscs/swissai/infra01/audio-datasets/processed/localized_narratives/<group>/
#
# Sharding policy (override via SHARD_ROWS):
#   default: shard every 50000 records → keeps individual parquet files < ~10 GB,
#   so the 507k-row open_images_train ends up as 11 shards instead of one
#   100 GB monolith.
#
# Submission patterns:
#   sbatch --array=0-9 pack_to_parquet.slurm                    # all 10 groups
#   sbatch --array=0,3 pack_to_parquet.slurm                    # only the two largest
#   LN_GROUPS="coco_val flickr30k_val" sbatch --array=0-1 pack_to_parquet.slurm
#   SHARD_ROWS=0 sbatch --array=0-9 pack_to_parquet.slurm        # single file per group
#
#SBATCH --account=infra01
#SBATCH --job-name=ln-parquet
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/localized_narratives/logs/%x-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/localized_narratives/logs/%x-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

DEFAULT_LN_GROUPS=(
    open_images_train
    open_images_validation
    open_images_test
    coco_train
    coco_val
    flickr30k_train
    flickr30k_val
    flickr30k_test
    ade20k_train
    ade20k_validation
)

# NOTE: do NOT use the bash variable name `GROUPS` — it's a readonly builtin
# (supplementary group IDs) and gets pre-set to a numeric value like "65603".
if [[ -n "${LN_GROUPS:-}" ]]; then
    IFS=' ' read -ra GROUP_ARRAY <<< "${LN_GROUPS}"
else
    GROUP_ARRAY=("${DEFAULT_LN_GROUPS[@]}")
fi

if [[ "${SLURM_ARRAY_TASK_ID:-NONE}" == "NONE" ]]; then
    echo "ERROR: this script must be submitted as an array job (sbatch --array=...)" >&2
    exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= ${#GROUP_ARRAY[@]} )); then
    echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} out of range for ${#GROUP_ARRAY[@]} groups" >&2
    exit 1
fi

GROUP="${GROUP_ARRAY[$SLURM_ARRAY_TASK_ID]}"

SOURCE_ROOT="${SOURCE_ROOT:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/localized_narratives}"
OUT_ROOT="${OUT_ROOT:-/capstor/store/cscs/swissai/infra01/audio-datasets/processed/localized_narratives}"
SHARD_ROWS="${SHARD_ROWS:-50000}"
# 288-core node: 256 IO threads + headroom for the writer/main thread.
AUDIO_WORKERS="${AUDIO_WORKERS:-256}"
BATCH_ROWS="${BATCH_ROWS:-4000}"

SCRIPT_DIR="/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/localized_narratives"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
SCRIPT="${SCRIPT_DIR}/ln_to_parquet.py"

OUT_DIR="${OUT_ROOT}/${GROUP}"
mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "Localized Narratives → parquet"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Array task:    ${SLURM_ARRAY_TASK_ID}"
echo "Node:          $(hostname)"
echo "Group:         ${GROUP}"
echo "Source root:   ${SOURCE_ROOT}"
echo "Out dir:       ${OUT_DIR}"
echo "Shard rows:    ${SHARD_ROWS} (0 = single file)"
echo "Audio workers: ${AUDIO_WORKERS}"
echo "Started:       $(date)"
echo "=========================================="

"${PYTHON_BIN}" "${SCRIPT}" \
    --source-root "${SOURCE_ROOT}" \
    --output-dir  "${OUT_DIR}" \
    --group       "${GROUP}" \
    --shard-rows  "${SHARD_ROWS}" \
    --audio-workers "${AUDIO_WORKERS}" \
    --batch-rows  "${BATCH_ROWS}"

echo "=========================================="
echo "Done: ${GROUP} ($(date))"
ls -lh "${OUT_DIR}"
echo "=========================================="
