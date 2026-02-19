#!/bin/bash
# Download & extract Libri-Light unsupervised dataset using aria2c
#
# Usage:
#   ./download_libri_light.sh small medium large
#   ./download_libri_light.sh all
#   ./download_libri_light.sh large    # 51,934 hours, 3.05 TB

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tools

DEST_DIR=${DEST_DIR:-/iopsstor/scratch/cscs/${USER}/audio-datasets/raw/libri-light}
CONNECTIONS=${CONNECTIONS:-16}
EXTRACT_JOBS=${EXTRACT_JOBS:-16}

BASE_URL="https://dl.fbaipublicfiles.com/librilight/data"

declare -A SPLITS=(
  [small]="${BASE_URL}/small.tar"           # 577h, 35 GB
  [medium]="${BASE_URL}/medium.tar"         # 5,193h, 321 GB
  [large]="${BASE_URL}/large.tar"           # 51,934h, 3.05 TB
)

# Parse arguments
if [ $# -eq 0 ]; then
  echo "Usage: $0 <split...> | all"
  echo ""
  echo "Available splits:"
  echo "  small       577 hours, 35 GB"
  echo "  medium      5,193 hours, 321 GB"
  echo "  large       51,934 hours, 3.05 TB"
  echo ""
  echo "  all         downloads small + medium + large (unlab-60k)"
  exit 0
fi

REQUESTED=("$@")
if [ "${REQUESTED[0]}" = "all" ]; then
  REQUESTED=(small medium large)
fi

mkdir -p "${DEST_DIR}"
TARBALL_DIR="${DEST_DIR}/_tarballs"
mkdir -p "${TARBALL_DIR}"

echo "========================================"
echo "Libri-Light Download"
echo "Destination: ${DEST_DIR}"
echo "Splits: ${REQUESTED[*]}"
echo "========================================"

# ---------------------------------------------------------------
# Step 1: Generate URL list & download
# ---------------------------------------------------------------
echo ""
echo "[$(date '+%F %T')] Step 1: Downloading tarballs..."

URL_FILE="${TARBALL_DIR}/urls.txt"
SESSION_FILE="${TARBALL_DIR}/aria2_session.txt"
> "${URL_FILE}"

for split in "${REQUESTED[@]}"; do
  if [ -z "${SPLITS[$split]+x}" ]; then
    echo "  Unknown split: ${split}, skipping"
    continue
  fi
  if [ -d "${DEST_DIR}/${split}" ]; then
    echo "  [${split}] Already extracted, skipping"
    continue
  fi
  echo "${SPLITS[$split]}" >> "${URL_FILE}"
done

TOTAL_URLS=$(wc -l < "${URL_FILE}")
if [ "${TOTAL_URLS}" -eq 0 ]; then
  echo "  Nothing to download"
  exit 0
fi
echo "  ${TOTAL_URLS} tarball(s) to download"

if [ -f "${SESSION_FILE}" ] && [ -s "${SESSION_FILE}" ]; then
    INPUT="${SESSION_FILE}"
    echo "  Resuming from session file"
else
    INPUT="${URL_FILE}"
fi

aria2c \
    --input-file="${INPUT}" \
    --save-session="${SESSION_FILE}" \
    --dir="${TARBALL_DIR}" \
    --max-concurrent-downloads=4 \
    --max-connection-per-server="${CONNECTIONS}" \
    --split="${CONNECTIONS}" \
    --min-split-size=100M \
    --continue=true \
    --auto-file-renaming=false \
    --max-tries=5 \
    --retry-wait=10 \
    --summary-interval=60 \
    --console-log-level=notice

echo "[$(date '+%F %T')] Step 1 done"

# ---------------------------------------------------------------
# Step 2: Extract tarballs in parallel, then clean up
# ---------------------------------------------------------------
echo ""
echo "[$(date '+%F %T')] Step 2: Extracting..."

find "${TARBALL_DIR}" -name '*.tar' | \
    xargs -P "${EXTRACT_JOBS}" -I {} bash -c 'echo "  Extracting $(basename {})..." && tar xf {} -C "'"${DEST_DIR}"'" && rm {}'

echo "[$(date '+%F %T')] Step 2 done"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "========================================"
echo "Download & extraction complete"
for split in "${REQUESTED[@]}"; do
  if [ -d "${DEST_DIR}/${split}" ]; then
    size=$(du -sh "${DEST_DIR}/${split}" | cut -f1)
    echo "  ${split}: ${size}"
  fi
done
echo "========================================"
