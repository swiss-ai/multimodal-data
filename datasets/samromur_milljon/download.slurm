#!/bin/bash
# Download language-and-voice-lab/samromur_milljon (~50GB, Icelandic, CC-BY-4.0)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-samromur-mj
#SBATCH --reservation=SD-69241-apertus-1-5-6
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/samromur-mj-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/samromur-mj-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288

set -euo pipefail

DATASET_REPO="language-and-voice-lab/samromur_milljon"
REVISION="dca915c25bcab234662e8b5d2ba77b3cf55d1703"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___language-and-voice-lab___samromur_milljon"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "Samromur Milljon download"
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
