#!/bin/bash
# Download audiocite.net French speech corpus from OpenSLR 139
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-audiocite
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/audiocite-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/audiocite-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tools

DEST_DIR=${DEST_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/audiocite}
BASE_URL="https://www.openslr.org/resources/139"
CONNECTIONS=16

mkdir -p "${DEST_DIR}"

echo "========================================"
echo "Audiocite.net Download (OpenSLR 139)"
echo "Dest: ${DEST_DIR}"
echo "========================================"

# Build URL list: metadata + md5 + 34 audio zips
URL_FILE=$(mktemp)
echo "${BASE_URL}/audiocite.net_0.zip" >> "${URL_FILE}"
echo "${BASE_URL}/audiocite.net_md5.txt" >> "${URL_FILE}"
for i in $(seq 1 34); do
  echo "${BASE_URL}/audiocite.net_${i}.zip" >> "${URL_FILE}"
done

echo "Downloading $(wc -l < "${URL_FILE}") files..."

aria2c -x "${CONNECTIONS}" -s "${CONNECTIONS}" \
  --continue=true \
  --max-tries=5 \
  --retry-wait=10 \
  --max-concurrent-downloads=4 \
  --summary-interval=60 \
  -d "${DEST_DIR}" \
  -i "${URL_FILE}"

rm -f "${URL_FILE}"

# Verify checksums
echo "Verifying checksums..."
cd "${DEST_DIR}"
if md5sum -c audiocite.net_md5.txt; then
  echo "All checksums OK"
else
  echo "WARNING: Some checksums failed"
fi

echo "[$(date '+%F %T')] Download finished"
echo "Done."
