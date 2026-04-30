#!/bin/bash
# Download MediaSpeech (OpenSLR 108) — 4 languages x 10h each (~2.4GB, CC-BY-4.0)
# AR, ES, FR, TR broadcast media speech.
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-mediaspeech
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mediaspeech-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mediaspeech-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=SD-69241-apertus-1-5-5

set -euo pipefail

DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/openslr108___mediaspeech"
BASE_URL="https://openslr.trmal.net/resources/108"
CONNECTIONS=16

mkdir -p "${DEST_DIR}"

echo "========================================"
echo "MediaSpeech (OpenSLR 108) Download"
echo "Dest: ${DEST_DIR}"
echo "========================================"

URL_FILE=$(mktemp)
for lang in AR ES FR TR; do
    echo "${BASE_URL}/${lang}.tgz" >> "${URL_FILE}"
done

aria2c -x "${CONNECTIONS}" -s "${CONNECTIONS}" \
    --continue=true \
    --max-tries=10 \
    --retry-wait=10 \
    --summary-interval=30 \
    -d "${DEST_DIR}" \
    -i "${URL_FILE}"

rm -f "${URL_FILE}"

# Extract each language archive
for lang in AR ES FR TR; do
    if [ -f "${DEST_DIR}/${lang}.tgz" ] && [ ! -d "${DEST_DIR}/${lang}" ]; then
        echo "Extracting ${lang}.tgz..."
        tar --no-same-owner --no-same-permissions -xzf "${DEST_DIR}/${lang}.tgz" -C "${DEST_DIR}/"
    fi
done

echo "[$(date '+%F %T')] Done"
echo "Final size:"
du -sh "${DEST_DIR}"
