#!/bin/bash
# Download HUI-Audio-Corpus-German (clean subset) from opendata.iisys.de
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-hui-de
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/hui-de-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/hui-de-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tools

DEST_DIR=${DEST_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hui-audio-corpus-german}
BASE_URL="https://opendata.iisys.de/opendata/Datasets/HUI-Audio-Corpus-German/dataset_clean"
CONNECTIONS=16

mkdir -p "${DEST_DIR}"

echo "========================================"
echo "HUI-Audio-Corpus-German (Clean) Download"
echo "Dest: ${DEST_DIR}"
echo "========================================"

URL_FILE=$(mktemp)
for name in Bernd_Ungerer_Clean Hokuspokus_Clean Friedrich_Clean Karlsson_Clean Eva_K_Clean others_Clean; do
  echo "${BASE_URL}/${name}.zip" >> "${URL_FILE}"
done

echo "Downloading $(wc -l < "${URL_FILE}") files..."

aria2c -x "${CONNECTIONS}" -s "${CONNECTIONS}" \
  --continue=true \
  --max-tries=5 \
  --retry-wait=10 \
  --max-concurrent-downloads=6 \
  --summary-interval=30 \
  -d "${DEST_DIR}" \
  -i "${URL_FILE}"

rm -f "${URL_FILE}"

echo "[$(date '+%F %T')] Download finished"

# Extract all zip files
echo "Extracting zip files..."
for f in "${DEST_DIR}"/*.zip; do
  echo "  Extracting $(basename "$f")..."
  unzip -o -q "$f" -d "${DEST_DIR}/"
done

echo "[$(date '+%F %T')] Extraction finished"
echo "Done."
