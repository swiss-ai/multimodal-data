#!/bin/bash
# Download NB Tale Norwegian speech corpus from Språkbanken
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-nb-tale
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/nb-tale-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/nb-tale-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tools

DEST_DIR=${DEST_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/nb-tale}
BASE_URL="https://www.nb.no/sbfil/talegjenkjenning/lingit"
CONNECTIONS=16

mkdir -p "${DEST_DIR}"

echo "========================================"
echo "NB Tale Norwegian Speech Download"
echo "Dest: ${DEST_DIR}"
echo "========================================"

# Build URL list
URL_FILE=$(mktemp)
for f in sennheiser_1 sennheiser_2 sennheiser_3 shure_1 shure_2 shure_3; do
  echo "${BASE_URL}/${f}.tar.gz" >> "${URL_FILE}"
done
echo "${BASE_URL}/Annotation.zip" >> "${URL_FILE}"
echo "${BASE_URL}/Documentation.zip" >> "${URL_FILE}"
echo "${BASE_URL}/01_NB_Tale_teknisk.pdf" >> "${URL_FILE}"

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

# Extract annotations and documentation
echo "Extracting Annotation.zip..."
unzip -o -q "${DEST_DIR}/Annotation.zip" -d "${DEST_DIR}/"
echo "Extracting Documentation.zip..."
unzip -o -q "${DEST_DIR}/Documentation.zip" -d "${DEST_DIR}/"

# Extract audio tarballs in parallel
echo "Extracting audio tarballs..."
ls "${DEST_DIR}"/*.tar.gz | xargs -P 6 -I {} tar --no-same-owner --no-same-permissions -xzf {} -C "${DEST_DIR}/"

echo "[$(date '+%F %T')] Extraction finished"
echo "Done."
