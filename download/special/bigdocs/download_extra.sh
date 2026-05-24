#!/usr/bin/env bash
# Downloads external image archives required by the BigDocs-7.5M dataset.
#
# BigDocs-7.5M (huggingface.co/datasets/ServiceNow/BigDocs-7.5M) references
# images from three external sources that must be downloaded separately:
#
#   - COCO train2014        (http://images.cocodataset.org)
#   - TextVQA train images  (https://dl.fbaipublicfiles.com)
#   - TableFact             (https://tablefact.s3-us-west-2.amazonaws.com)
#
# Usage:
#   OUTPUT_DIR=/path/to/bigdocs/subsets bash download_extra.sh

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/data/bigdocs/subsets}"
mkdir -p "$OUTPUT_DIR"

wget -c --timeout=60 --waitretry=10 --tries=5 \
    -P "$OUTPUT_DIR" \
    "http://images.cocodataset.org/zips/train2014.zip"

wget -c --timeout=60 --waitretry=10 --tries=5 \
    -P "$OUTPUT_DIR" \
    "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"

wget -c --timeout=60 --waitretry=10 --tries=5 \
    -P "$OUTPUT_DIR" \
    "https://tablefact.s3-us-west-2.amazonaws.com/preprocessed_data_program.zip"

echo "Download complete. Unzip each archive:"
echo "  for f in ${OUTPUT_DIR}/*.zip; do unzip \$f -d ${OUTPUT_DIR}; done"
