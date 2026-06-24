#!/bin/bash
# Download BIFOLD-BigEarthNetv2-0/BigEarthNet.txt (text annotations from HF, ~0.5GB)
# + BigEarthNet-S1 (SAR, ~51GB) and BigEarthNet-S2 (multispectral, ~59GB) from Zenodo
#SBATCH --account=infra01
#SBATCH --job-name=dl-bigearthnet
#SBATCH --reservation=SD-69241-apertus-1-5-5
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/bigearthnet_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/bigearthnet_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# ---- 1. Text annotations from HuggingFace ----
DATASET_REPO="BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"
REVISION="72d865f2146f"
DEST="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___BIFOLD-BigEarthNetv2-0___BigEarthNet"
TEXT_DEST="${DEST}/text"
mkdir -p "${TEXT_DEST}"
export HF_HUB_CACHE="${TEXT_DEST}/.hf_cache"

echo "=========================================="
echo "[1/3] Downloading text annotations: ${DATASET_REPO}"
echo "=========================================="

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${TEXT_DEST}" \
    --max-workers 64

rm -rf "${HF_HUB_CACHE}"

CONNECTIONS=16

# ---- 2. Sentinel-2 multispectral images from Zenodo ----
echo "=========================================="
echo "[2/3] Downloading Sentinel-2 (~59 GiB)"
echo "=========================================="

aria2c -x "$CONNECTIONS" -s "$CONNECTIONS" \
    --continue=true \
    --max-tries=10 \
    --retry-wait=10 \
    --summary-interval=30 \
    -d "${DEST}" \
    -o "BigEarthNet-S2.tar.zst" \
    "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst"

# ---- 3. Sentinel-1 SAR images from Zenodo ----
echo "=========================================="
echo "[3/3] Downloading Sentinel-1 (~51 GiB)"
echo "=========================================="

aria2c -x "$CONNECTIONS" -s "$CONNECTIONS" \
    --continue=true \
    --max-tries=10 \
    --retry-wait=10 \
    --summary-interval=30 \
    -d "${DEST}" \
    -o "BigEarthNet-S1.tar.zst" \
    "https://zenodo.org/records/10891137/files/BigEarthNet-S1.tar.zst"

echo "=========================================="
echo "Done: bigearthnet"
du -sh "${DEST}"
