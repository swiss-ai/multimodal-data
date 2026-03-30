#!/bin/bash
# Download UCSC-VLAA/Recap-DataComp-1B (~1.88B rows, 1.05TB parquets, image URLs + recaptions, CC BY 4.0)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-recap-datacomp-1b
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/recap-datacomp-1b-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/recap-datacomp-1b-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="UCSC-VLAA/Recap-DataComp-1B"
REVISION="457c44d98651bcfdfb3cc8695f5e60a0d2705e78"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/UCSC-VLAA___Recap-DataComp-1B"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "Recap-DataComp-1B download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --revision "${REVISION}" \
  --local-dir "${DEST_DIR}" \
  --max-workers 64

echo "[$(date '+%F %T')] Download finished"
echo "Pinned to commit: ${REVISION}"

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "Done."
