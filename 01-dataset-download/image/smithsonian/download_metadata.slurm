#!/bin/bash
#SBATCH --job-name=smithsonian-metadata
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_26_02
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=288
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/smithsonian-metadata-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/smithsonian-metadata-%j.err

# Smithsonian Open Access — all 37 collections, metadata only (line-delimited JSON).
# License: CC0-1.0 (public domain).
# Bucket is public; --no-sign-request avoids needing AWS credentials.
#
# This pulls JSONL metadata only. Images live on the si.edu CDN and are
# referenced in each record under content.descriptiveNonRepeating.online_media.
# A follow-up script will parse the metadata and download the image bytes.

set -euo pipefail

DEST=/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/s3___smithsonian___OpenAccess/metadata/edan
mkdir -p "$DEST"

export PATH="/users/xyixuan/.local/aarch64/bin:$PATH"

aws --version

# Tune s3 concurrency for the 288-core node. Default is 10; 64 lets us
# saturate the egress without tripping aws-cli's own backoff logic.
aws configure set default.s3.max_concurrent_requests 64
aws configure set default.s3.max_queue_size 2000
aws configure set default.s3.multipart_threshold 64MB

echo "=== Sync start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
time aws s3 sync --no-sign-request \
    s3://smithsonian-open-access/metadata/edan/ \
    "$DEST/" \
    --cli-read-timeout 0 \
    --cli-connect-timeout 60 \
    --only-show-errors

echo "=== Sync done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo
echo "=== Summary ==="
du -sh "$DEST"
find "$DEST" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort | while read col; do
    n=$(find "$DEST/$col" -name '*.txt' 2>/dev/null | wc -l)
    sz=$(du -sh "$DEST/$col" 2>/dev/null | cut -f1)
    printf "  %-22s %5d files  %s\n" "$col" "$n" "$sz"
done
