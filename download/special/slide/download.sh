#!/usr/bin/env bash
# Downloads the SLIDE histopathology dataset from Figshare.
# Article DOI: https://doi.org/10.6084/m9.figshare.26172919
#
# The archive contains train.zip, val.zip, and test.zip.
# After download, unzip each archive into OUTPUT_DIR.
#
# Usage:
#   OUTPUT_DIR=/path/to/slide bash download.sh

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/data/slide}"
mkdir -p "$OUTPUT_DIR"

wget -c --timeout=120 --waitretry=10 --tries=5 \
    -P "$OUTPUT_DIR" \
    "https://figshare.com/ndownloader/articles/26172919/versions/2" \
    -O "${OUTPUT_DIR}/slide_dataset.zip"

echo "Download complete. Unzip with:"
echo "  unzip ${OUTPUT_DIR}/slide_dataset.zip -d ${OUTPUT_DIR}"
