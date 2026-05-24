#!/bin/bash
# Pack dataset directories into squashfs images using mksquashfs.
#
# Usage:
#   ./pack_squashfs.sh SOURCE_DIR DEST_DIR [PARALLEL_JOBS] [CORES_PER_JOB]
#
# Example:
#   ./pack_squashfs.sh /data/my-dataset /data/archive 12 20
#
# Each subdirectory under SOURCE_DIR becomes a separate .sqfs file in DEST_DIR.
# Processing is batched: PARALLEL_JOBS (default 12) at a time, each using
# CORES_PER_JOB (default 20) threads.

set -euo pipefail

SOURCE_DIR="${1:?Usage: $0 SOURCE_DIR DEST_DIR [PARALLEL_JOBS] [CORES_PER_JOB]}"
DEST_DIR="${2:?Usage: $0 SOURCE_DIR DEST_DIR [PARALLEL_JOBS] [CORES_PER_JOB]}"
PARALLEL_JOBS="${3:-12}"
CORES_PER_JOB="${4:-20}"

mkdir -p "$DEST_DIR"

echo "Packing subdirectories of $SOURCE_DIR into $DEST_DIR"
echo "Parallel jobs: $PARALLEL_JOBS, cores per job: $CORES_PER_JOB"

count=0
for part in "$SOURCE_DIR"/*/; do
    dir_name=$(basename "$part")
    src_path="$part"
    out_file="$DEST_DIR/${dir_name}.sqfs"

    echo "  Packing $dir_name ..."
    mksquashfs "$src_path" "$out_file" \
        -noI -noD -noF -noX -no-fragments \
        -processors "$CORES_PER_JOB" > /dev/null &

    ((count++))

    if (( count % PARALLEL_JOBS == 0 )); then
        echo "  Waiting for batch of $PARALLEL_JOBS to finish..."
        wait
    fi
done

echo "Waiting for final jobs to finish..."
wait
echo "All done — packed $count directories."
