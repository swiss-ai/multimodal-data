#!/bin/bash
# Download NaiveNeuron/SloPalSpeech (~61GB, 2806h Slovak parliament, CC-BY-4.0)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-slopalspeech
#SBATCH --reservation=SD-69241-apertus-1-5-6
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/slopalspeech-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/slopalspeech-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288

set -euo pipefail

DATASET_REPO="NaiveNeuron/SloPalSpeech"
REVISION="c8a0ee1733fc1f28dee7fb80fec24a2153280569"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___NaiveNeuron___SloPalSpeech"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "SloPalSpeech download"
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

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "Done."
