#!/bin/bash
# Download deepglint/DanQing (100M Chinese image-text pairs, ~12TB, CC BY 4.0)
# NOTE: This dataset triggered CSCS security alerts during previous ingestion.
#       Run URL safety filter BEFORE downloading images from URLs in this dataset.
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-danqing
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/danqing-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/danqing-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="deepglint/DanQing"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/ms___deepglint___DanQing"

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export MODELSCOPE_CACHE="${DEST_DIR}/.ms_cache"

echo "========================================"
echo "DanQing download"
echo "Repo:     ${DATASET_REPO}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

# Read token from saved credentials
MS_TOKEN=$(python3 -c "from modelscope.hub.api import HubApi; print(HubApi()._get_token())" 2>/dev/null || cat ~/.modelscope/token 2>/dev/null || echo "")

modelscope download \
  --dataset "${DATASET_REPO}" \
  --local_dir "${DEST_DIR}" \
  --revision master \
  --token "${MS_TOKEN}" \
  --max-workers 64

echo "[$(date '+%F %T')] Download finished"

rm -rf "${MODELSCOPE_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "Done."
