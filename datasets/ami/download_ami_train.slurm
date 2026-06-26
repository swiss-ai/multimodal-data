#!/bin/bash
# Download edinburghcstr/ami train split only (ihm + sdm parquet shards).
# Uses a temporary HF cache under DEST_DIR and deletes it after success
# to avoid duplicate storage.
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-ami-train
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/ami-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/ami-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="${DATASET_REPO:-edinburghcstr/ami}"
REVISION="${REVISION:-46f28f2}"  # main as of 2026-03-03 (parquet layout)
DEST_DIR="${DEST_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/ami}"

mkdir -p "${DEST_DIR}"
uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1

if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.hf-token" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "${HOME}/.hf-token")"
fi

echo "========================================"
echo "AMI train-only download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --revision "${REVISION}" \
  --include "ihm/train-*.parquet" "sdm/train-*.parquet" "README.md" ".gitattributes" \
  --local-dir "${DEST_DIR}"

echo "[$(date '+%F %T')] Download finished"
echo "Pinned to revision: ${REVISION}"

# Clean up HF hub cache (duplicate of downloaded files)
echo "Cleaning up HF hub cache..."
rm -rf "${HF_HUB_CACHE}"
echo "Done."
