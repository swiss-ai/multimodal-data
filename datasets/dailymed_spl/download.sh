#!/usr/bin/env bash
# Download a single DailyMed SPL partition ZIP. Designed to be called from a
# SLURM array job (one array task = one partition = one node), but works
# standalone too.
#
# Env:
#   PARTITION  partition basename without .zip (e.g. dm_spl_release_human_rx_part1)
#   DEST       output root (default: /capstor/store/.../medical-datasets/raw/dailymed_spl)
#
# Source: https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm
# License: SPL text is manufacturer-authored; FDA / NLM distribute under
# regulatory authority. For commercial training, merger + fact/idea doctrines
# + zero enforcement history make label text + scientific images effectively
# usable. Avoid branded packaging photography in downstream filtering.
#
# Resume: aria2c --continue=true + --allow-overwrite=false skips complete files.

set -euo pipefail

DEST="${DEST:-/capstor/store/cscs/swissai/infra01/medical-datasets/raw/dailymed_spl}"
RAW_DIR="$DEST/raw_zips"
mkdir -p "$RAW_DIR"

PARTITION="${PARTITION:-}"
if [ -z "$PARTITION" ]; then
  echo "ERROR: PARTITION env var not set. e.g. PARTITION=dm_spl_release_human_rx_part1"
  exit 1
fi

BASE="https://dailymed-data.nlm.nih.gov/public-release-files"
URL="${BASE}/${PARTITION}.zip"
OUT="${RAW_DIR}/${PARTITION}.zip"

if [ -f "$OUT" ]; then
  echo "[${PARTITION}] already exists ($(du -h "$OUT" | cut -f1)), skipping"
  exit 0
fi

echo "[${PARTITION}] downloading ${URL}"
echo "                  -> ${OUT}"
aria2c \
  -x 16 -s 16 \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  --continue=true \
  --console-log-level=warn \
  --summary-interval=30 \
  --retry-wait=10 \
  --max-tries=5 \
  -d "$RAW_DIR" \
  -o "${PARTITION}.zip" \
  "$URL"

echo "[${PARTITION}] done. size: $(du -h "$OUT" | cut -f1)"
