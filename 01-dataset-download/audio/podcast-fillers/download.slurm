#!/bin/bash
# Download ylacombe/podcast_fillers_processed (51h English podcast fillers, CC-BY-3.0)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-podfillers
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/podfillers-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/podfillers-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="ylacombe/podcast_fillers_processed"
REVISION="7c0ecc9539330e289f74e53d48d5c6caf948ea27"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___ylacombe___podcast_fillers_processed"

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

echo "========================================"
echo "Podcast Fillers Download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

mkdir -p "${DEST_DIR}"

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
