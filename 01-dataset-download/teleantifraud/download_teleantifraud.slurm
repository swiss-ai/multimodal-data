#!/bin/bash
# Download JimmyMa99/TeleAntiFraud-28k (28K Chinese telephone anti-fraud speech, 13.3GB, Apache 2.0)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-teleantifraud
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/teleantifraud-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/teleantifraud-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="JimmyMa99/TeleAntiFraud-28k"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/ms___JimmyMa99___TeleAntiFraud-28k"

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export MODELSCOPE_CACHE="${DEST_DIR}/.ms_cache"

# Requires ModelScope login — run once interactively:
#   PYTHONPATH=/capstor/store/cscs/swissai/infra01/MLLM/pip-packages \
#   PATH=/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:$PATH \
#   modelscope login --token YOUR_TOKEN

echo "========================================"
echo "TeleAntiFraud-28k download"
echo "Repo:     ${DATASET_REPO}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

modelscope download \
  --dataset "${DATASET_REPO}" \
  --local_dir "${DEST_DIR}" \
  --max-workers 64

echo "[$(date '+%F %T')] Download finished"

rm -rf "${MODELSCOPE_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "Done."
