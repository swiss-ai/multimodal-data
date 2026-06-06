#!/bin/bash
# Download allenai/Molmo2-MultiImagePoint (~200 MiB, ODC-BY).
# Multi-image pointing / grounding SFT for Molmo 2. 95 parquet shards.
# NOTE: parquets are text-only (image references, not bytes). Downstream step must
# resolve the image refs to actual pixels — see README for the source-image map.
#
# Usage:
#   sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/molmo2_multiimagepoint/download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-molmo2-multiimagepoint
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/molmo2-multiimagepoint-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/molmo2-multiimagepoint-%x-%A.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=normal

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_REPO="allenai/Molmo2-MultiImagePoint"
REVISION="ab3d86ad22917698c24680138f20ff65693f402d"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___allenai___Molmo2-MultiImagePoint"
SCRIPT_PATH="01-dataset-download/image/molmo2_multiimagepoint/download.slurm"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

mkdir -p "${DEST_DIR}"

echo "========================================"
echo "Molmo2-MultiImagePoint Download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "Expected: ~200 MiB, ODC-BY, 95 parquet shards (text only — image refs only)"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${DEST_DIR}" \
    --max-workers 32

echo "[$(date '+%F %T')] Download finished"

cat > "${DEST_DIR}/_SOURCE.json" <<EOF
{
  "dataset_id": "hf___allenai___Molmo2-MultiImagePoint",
  "repo_url": "https://huggingface.co/datasets/${DATASET_REPO}",
  "revision": "${REVISION}",
  "license": "odc-by",
  "data_type": "sft",
  "task": "multi_image_pointing_grounding",
  "download_method": "huggingface-cli download with HF_HUB_ENABLE_HF_TRANSFER=1",
  "downloaded_at_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "script": "${SCRIPT_PATH}",
  "size_gib": 0.2,
  "notes": "Multi-image pointing / grounding SFT for Molmo 2. 95 parquet shards under parquet/. IMPORTANT: parquets carry text + image references (URLs / doc IDs), not embedded bytes. A separate image-resolution step is required before training. Complements single-image MolmoPoint-GUISyn."
}
EOF

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "[$(date '+%F %T')] Done"
