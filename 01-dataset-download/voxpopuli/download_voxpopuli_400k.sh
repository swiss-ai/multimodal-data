#!/bin/bash
# Download the remaining 300k hours of VoxPopuli (400k - existing 100k).
#
# Existing 100k lives on capstor and will NOT be re-downloaded:
#   /capstor/store/cscs/swissai/infra01/audio-datasets/voxpopuli/raw_audios/{lang}/{year}/*.ogg
#
# This script downloads only what's missing to iopsstor:
#   - _2 split tarballs for all 23 languages (the additional ~300k hours)
#   - Base tarballs for 8 languages not in the original download (bg, da, el, lv, lt, mt, pt, sv)
#
# Both the existing capstor 100k and this new iopsstor 300k can be tokenized
# into shars at the same capstor output directory.
#
# Supports resume: re-run the same command and aria2c picks up where it left off.
#
# Usage:
#   sbatch download_voxpopuli_400k.slurm
#   ./download_voxpopuli_400k.sh

set -euo pipefail

DEST_DIR=${DEST_DIR:-/iopsstor/scratch/cscs/${USER}/audio-datasets/raw/voxpopuli}
CONNECTIONS=${CONNECTIONS:-16}
CONCURRENT=${CONCURRENT:-16}
EXTRACT_JOBS=${EXTRACT_JOBS:-16}

BASE_URL="https://dl.fbaipublicfiles.com/voxpopuli/audios"

ALL_LANGS=(en de fr es pl it ro hu cs nl fi hr sk sl et lt pt bg el lv mt sv da)
NEW_LANGS=(lt pt bg el lv mt sv da)

mkdir -p "${DEST_DIR}"
TARBALL_DIR="${DEST_DIR}/_tarballs"
mkdir -p "${TARBALL_DIR}"

echo "========================================"
echo "VoxPopuli 400k - Download missing 300k"
echo "Destination: ${DEST_DIR}"
echo "========================================"

# ---------------------------------------------------------------
# Step 1: Generate URL list & download
# ---------------------------------------------------------------
echo ""
echo "[$(date '+%F %T')] Step 1: Downloading missing tarballs..."

URL_FILE="${TARBALL_DIR}/urls.txt"
SESSION_FILE="${TARBALL_DIR}/aria2_session.txt"
> "${URL_FILE}"

# _2 tarballs for all 23 languages (the additional ~300k hours)
for lang in "${ALL_LANGS[@]}"; do
    for year in $(seq 2009 2020); do
        echo "${BASE_URL}/${lang}_${year}_2.tar" >> "${URL_FILE}"
    done
done

# Base tarballs for 8 languages not in the original 100k download
for lang in "${NEW_LANGS[@]}"; do
    for year in $(seq 2009 2020); do
        echo "${BASE_URL}/${lang}_${year}.tar" >> "${URL_FILE}"
    done
done

TOTAL_URLS=$(wc -l < "${URL_FILE}")
echo "  ${TOTAL_URLS} tarballs to download"

# On re-run, prefer session file (tracks what's already done) over full URL list
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
    --max-concurrent-downloads="${CONCURRENT}" \
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
echo "[$(date '+%F %T')] Step 2: Extracting tarballs (${EXTRACT_JOBS} parallel jobs)..."

find "${TARBALL_DIR}" -name '*.tar' | \
    xargs -P "${EXTRACT_JOBS}" -I {} bash -c 'tar --no-same-owner --no-same-permissions -xf {} -C "'"${DEST_DIR}"'" && rm {}'

echo "[$(date '+%F %T')] Step 2 done"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "========================================"
echo "Download & extraction complete"
echo ""
for lang in $(ls -1 "${DEST_DIR}" | grep -v '^_' | sort); do
    if [ -d "${DEST_DIR}/${lang}" ]; then
        count=$(find "${DEST_DIR}/${lang}" -name '*.ogg' | wc -l)
        printf "  %-5s %'d files\n" "${lang}" "${count}"
    fi
done
echo "========================================"
