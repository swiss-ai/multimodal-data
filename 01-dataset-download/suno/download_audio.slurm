#!/bin/bash
# Download Suno MP3s from CDN using aria2c
# Requires urls.txt from download_metadata.slurm
#SBATCH --account=infra01
#SBATCH --job-name=suno-audio
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/suno-audio-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/suno-audio-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tools

DEST_DIR=${DEST_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/suno}
CLIPS_DIR="${DEST_DIR}/clips"
URLS_FILE="${DEST_DIR}/urls.txt"

if [ ! -f "${URLS_FILE}" ]; then
    echo "ERROR: ${URLS_FILE} not found. Run download_metadata.slurm first."
    exit 1
fi

mkdir -p "${CLIPS_DIR}"

echo "========================================"
echo "Suno Audio Download"
echo "URLs file: ${URLS_FILE}"
echo "Dest: ${CLIPS_DIR}"
echo "Total URLs: $(grep -c '^http' "${URLS_FILE}")"
echo "========================================"

echo "[$(date '+%F %T')] Starting aria2c download..."
aria2c \
    -i "${URLS_FILE}" \
    -d "${CLIPS_DIR}" \
    -j 16 \
    -x 2 \
    --max-overall-download-limit=100M \
    --auto-file-renaming=false \
    --continue=true \
    --max-tries=5 \
    --retry-wait=5 \
    --timeout=30 \
    --log="${DEST_DIR}/aria2c.log" \
    --log-level=warn \
    --summary-interval=60

echo "[$(date '+%F %T')] Download finished"
echo "Files downloaded: $(ls "${CLIPS_DIR}" | wc -l)"
