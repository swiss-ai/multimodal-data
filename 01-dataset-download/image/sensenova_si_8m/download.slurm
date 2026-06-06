#!/bin/bash
# Download sensenova/SenseNova-SI-8M (~2.4 TiB, Apache 2.0, English VQA, 8M samples).
# 52 image_part_XXX.zip shards + SenseNova-SI-8M.jsonl + SenseNova-SI-8M.parquet + manifest.txt.
# Parallelized as a 6-task array; first 4 tasks pull 9 shards each, last 2 pull 8 shards each
# (52 = 4*9 + 2*8). Task 0 additionally pulls the manifest / annotation files.
#
# Usage:
#   sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/sensenova_si_8m/download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-sensenova-si-8m
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/sensenova-si-8m-%x-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/sensenova-si-8m-%x-%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_REPO="sensenova/SenseNova-SI-8M"
REVISION="2f1c0b6136417f5e2423aff839086636858de3f0"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___sensenova___SenseNova-SI-8M"
SCRIPT_PATH="01-dataset-download/image/sensenova_si_8m/download.slurm"

# Each array task gets its own HF_HUB_CACHE so tasks don't fight over lock files
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache_task${SLURM_ARRAY_TASK_ID}"

mkdir -p "${DEST_DIR}"

# 52 zip parts (001..052), 6 tasks. 52 = 4*9 + 2*8, so first 4 tasks get 9 shards, last 2 get 8.
#   Task 0: 001-009  Task 1: 010-018  Task 2: 019-027
#   Task 3: 028-036  Task 4: 037-044  Task 5: 045-052
NUM_SHARDS=52
NUM_TASKS=6
BASE=$(( NUM_SHARDS / NUM_TASKS ))         # 8
EXTRA=$(( NUM_SHARDS % NUM_TASKS ))        # 4
if [ "${SLURM_ARRAY_TASK_ID}" -lt "${EXTRA}" ]; then
    PER_TASK=$(( BASE + 1 ))                # 9
    START=$(( SLURM_ARRAY_TASK_ID * PER_TASK + 1 ))
else
    PER_TASK=${BASE}                        # 8
    START=$(( EXTRA * (BASE + 1) + (SLURM_ARRAY_TASK_ID - EXTRA) * BASE + 1 ))
fi
END=$(( START + PER_TASK - 1 ))

# huggingface-cli's --include uses argparse nargs='+' with default store action,
# so multiple --include flags OVERWRITE; pass all patterns to a single --include.
INCLUDE_PATTERNS=()
for i in $(seq -f "%03g" "${START}" "${END}"); do
    INCLUDE_PATTERNS+=( "images_part_${i}.zip" )
done

# Task 0 also grabs the manifests and annotation files
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    INCLUDE_PATTERNS+=(
        "SenseNova-SI-8M.jsonl"
        "SenseNova-SI-8M.parquet"
        "SenseNova-SI-8M_1000samples.parquet"
        "manifest.txt"
        "extract_all.sh"
        "README.md"
        "README_CN.md"
    )
fi

echo "========================================"
echo "SenseNova-SI-8M Download (array task ${SLURM_ARRAY_TASK_ID}/${NUM_TASKS})"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Shards:   images_part_$(printf '%03d' ${START}).zip .. images_part_$(printf '%03d' ${END}).zip"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${DEST_DIR}" \
    --include "${INCLUDE_PATTERNS[@]}" \
    --max-workers 64

echo "[$(date '+%F %T')] Task ${SLURM_ARRAY_TASK_ID} download finished"
rm -rf "${HF_HUB_CACHE}"

# Task 0 writes the _SOURCE.json after its work is done.
# Note: it won't reflect whether tasks 1-3 succeeded; check log files for those.
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    cat > "${DEST_DIR}/_SOURCE.json" <<EOF
{
  "dataset_id": "hf___sensenova___SenseNova-SI-8M",
  "repo_url": "https://huggingface.co/datasets/${DATASET_REPO}",
  "revision": "${REVISION}",
  "license": "apache-2.0",
  "paper": "arxiv:2511.13719",
  "download_method": "huggingface-cli download, 6-task SLURM array striping 52 zip shards (first 4 tasks: 9 each; last 2 tasks: 8 each) + manifest files on task 0, HF_HUB_ENABLE_HF_TRANSFER=1",
  "downloaded_at_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "script": "${SCRIPT_PATH}",
  "size_gib": 2391,
  "shards": 52,
  "notes": "English VQA / spatial-intelligence SFT mix (8M samples). 52 zip parts of images + master JSONL + Parquet. Run extract_all.sh after download completes to unpack the zips."
}
EOF
    echo "[$(date '+%F %T')] _SOURCE.json written"
fi

echo "Final size so far:"
du -sh "${DEST_DIR}"
