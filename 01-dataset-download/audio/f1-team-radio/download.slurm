#!/bin/bash
# Download MikCil/f1-team-radio (14.7K clips, F1 radio comms, CC-BY-4.0)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-f1radio
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/f1radio-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/f1radio-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="MikCil/f1-team-radio"
REVISION="$(curl -s https://huggingface.co/api/datasets/MikCil/f1-team-radio | python3 -c 'import sys,json; print(json.load(sys.stdin)["sha"])')"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___MikCil___f1-team-radio"

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

echo "========================================"
echo "F1 Team Radio Download"
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
