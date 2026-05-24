#!/usr/bin/env bash
# Downloads the SkyScript dataset (satellite imagery + captions).
# Source: https://github.com/wangzhecheng/SkyScript
#
# Usage:
#   OUTPUT_DIR=/path/to/skyscript bash download.sh

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/data/skyscript}"
BASE_URL="https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript"

mkdir -p "$OUTPUT_DIR"

# Image archives (parts 2-7; part 1 is not available for download separately)
for i in 2 3 4 5 6 7; do
    wget -c --timeout=60 --waitretry=10 --tries=5 \
        -P "$OUTPUT_DIR" \
        "${BASE_URL}/images${i}.zip"
done

# Metadata CSV files
for i in 2 3 4 5 6 7; do
    wget -c --timeout=60 --waitretry=10 --tries=5 \
        -P "$OUTPUT_DIR" \
        "${BASE_URL}/meta${i}.zip"
done

# Full training set metadata (unfiltered 5M)
wget -c --timeout=60 --waitretry=10 --tries=5 \
    -P "$OUTPUT_DIR" \
    "${BASE_URL}/dataframe/SkyScript_train_unfiltered_5M.csv"

echo "Download complete. Unzip each archive:"
echo "  for f in ${OUTPUT_DIR}/*.zip; do unzip \$f -d ${OUTPUT_DIR}; done"
