#!/usr/bin/env bash
# Downloads MedMax from the HuggingFace Hub resolve endpoint.
# The dataset is distributed as split tar.gz archives.
#
# Usage:
#   OUTPUT_DIR=/path/to/medmax bash download.sh
#
# After download, reassemble:
#   cat images.tar.gz.a* | tar -xz -C "$OUTPUT_DIR"

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/data/medmax}"
BASE_URL="https://huggingface.co/datasets/mint-medmax/medmax_data/resolve/main"

mkdir -p "$OUTPUT_DIR"

parts=(aa ab ac ad ae af ag ah ai aj ak al am an ao ap aq)
for part in "${parts[@]}"; do
    wget -c --timeout=60 --waitretry=10 --tries=5 \
        -P "$OUTPUT_DIR" \
        "${BASE_URL}/images.tar.gz.${part}"
done

# The SLAKE subset is also included in the medmax repository
wget -c --timeout=60 --waitretry=10 --tries=5 \
    -P "$OUTPUT_DIR" \
    "${BASE_URL}/slake.tar.gz"

echo "Download complete. Reassemble with:"
echo "  cat ${OUTPUT_DIR}/images.tar.gz.a* | tar -xz -C ${OUTPUT_DIR}"
