#!/bin/bash
#SBATCH --job-name=smithsonian-images
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_26_02
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=288
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/smithsonian-images-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/smithsonian-images-%j.err

# Smithsonian Open Access — full-resolution JPEG masters from all 23 /media/
# collections. TIFF masters are excluded (same content as JPEGs at 2-4x the
# bytes, no new visual information for VLM training).
#
# License: CC0-1.0. Bucket is public; --no-sign-request avoids creds.
# robots.txt on ids.si.edu CDN is restrictive, but the S3 bucket is the
# bucket is the intended bulk-access channel (hosted on "Registry of Open
# Data on AWS" with no access restriction of its own).

set -euo pipefail

DEST=/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/s3___smithsonian___OpenAccess/media
mkdir -p "$DEST"

export PATH="/users/xyixuan/.local/aarch64/bin:$PATH"

aws --version

# Tune s3 concurrency for the 288-core node. Default is 10.
aws configure set default.s3.max_concurrent_requests 128
aws configure set default.s3.max_queue_size 4000
aws configure set default.s3.multipart_threshold 64MB

# Discover collections dynamically so we don't drift from the bucket.
COLLECTIONS=$(aws s3 ls --no-sign-request s3://smithsonian-open-access/media/ \
    | awk '/^ *PRE / {gsub("/","",$2); print $2}' \
    | grep -v '^$')

echo "=== Start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Collections to pull:"
echo "$COLLECTIONS" | sed 's/^/  /'
echo

for col in $COLLECTIONS; do
    echo "=== [$col] $(date -u +%H:%M:%S) ==="
    time aws s3 sync --no-sign-request \
        "s3://smithsonian-open-access/media/$col/" \
        "$DEST/$col/" \
        --exclude "*.tif" --exclude "*.tiff" \
        --cli-read-timeout 0 \
        --cli-connect-timeout 60 \
        --only-show-errors || echo "WARN: $col sync exited non-zero, continuing"
    sz=$(du -sh "$DEST/$col" 2>/dev/null | cut -f1)
    n=$(find "$DEST/$col" -type f 2>/dev/null | wc -l)
    echo "  [$col] done: $n files, $sz"
done

echo
echo "=== Final summary $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
du -sh "$DEST"
for col in $COLLECTIONS; do
    sz=$(du -sh "$DEST/$col" 2>/dev/null | cut -f1 || echo "-")
    n=$(find "$DEST/$col" -type f 2>/dev/null | wc -l)
    printf "  %-14s %8d files  %s\n" "$col" "$n" "$sz"
done
