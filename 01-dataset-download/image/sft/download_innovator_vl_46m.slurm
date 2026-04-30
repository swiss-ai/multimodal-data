#!/bin/bash
# Download InnovatorLab/Innovator-VL-Instruct-46M (~11.3TB, 15568 parquets)
# Split across 8 array tasks for parallel download.
#
# Usage:
#   sbatch --array=0-7 download_innovator_vl_46m.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dl-innovator
#SBATCH --reservation=SD-69241-apertus-1-5-3
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/innovator_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/innovator_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -uo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_REPO="InnovatorLab/Innovator-VL-Instruct-46M"
REVISION="1e7bac4db314"
DEST="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___InnovatorLab___Innovator-VL-Instruct-46M"
export HF_HUB_CACHE="${DEST}/.hf_cache"

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download

# Generate include patterns based on array task
# 15568 files split into 8 chunks. Use python to get the exact file list.
INCLUDE=$(python3 -c "
from huggingface_hub import HfApi
api = HfApi()
files = sorted(f for f in api.list_repo_files('${DATASET_REPO}', repo_type='dataset') if f.endswith('.parquet'))
n = len(files)
chunk = (n + 7) // 8
start = ${SLURM_ARRAY_TASK_ID} * chunk
end = min(start + chunk, n)
for f in files[start:end]:
    print(f)
" > /tmp/filelist_${SLURM_ARRAY_TASK_ID}.txt)

echo "=========================================="
echo "Downloading: ${DATASET_REPO} (chunk ${SLURM_ARRAY_TASK_ID}/8)"
echo "Files: $(wc -l < /tmp/filelist_${SLURM_ARRAY_TASK_ID}.txt)"
echo "=========================================="

# Download files one at a time, skipping already-downloaded ones
FAIL_COUNT=0
SKIP_COUNT=0
while read -r fname; do
    if [ -f "${DEST}/${fname}" ]; then
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi
    if ! huggingface-cli download "${DATASET_REPO}" "${fname}" \
        --repo-type dataset \
        --revision "${REVISION}" \
        --local-dir "${DEST}"; then
        echo "WARN: failed to download ${fname}, continuing..." >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done < /tmp/filelist_${SLURM_ARRAY_TASK_ID}.txt
echo "Skipped ${SKIP_COUNT} already-downloaded files, ${FAIL_COUNT} failures"
if [ "${FAIL_COUNT}" -gt 0 ]; then
    exit 1
fi

rm -f /tmp/filelist_${SLURM_ARRAY_TASK_ID}.txt
rm -rf "${HF_HUB_CACHE}"
echo "Done: chunk ${SLURM_ARRAY_TASK_ID}"
