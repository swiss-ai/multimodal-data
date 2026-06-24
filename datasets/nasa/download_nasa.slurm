#!/bin/bash
#SBATCH --job-name=nasa-images
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_26_02
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=288
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/nasa-images-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/nasa-images-%j.err

# NASA Image and Video Library — full catalog.
# License: PD (17 USC §105). Some images are NASA/ESA joint (Hubble/JWST) —
# classified at download time as CC-BY-SA-3.0-IGO via center/creator fields.
# robots.txt on all relevant NASA domains is permissive or absent.

set -euo pipefail

DEST=/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/web___nasa___images
SRC=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/nasa
mkdir -p "$DEST"

cd "$SRC"

echo "=== Phase 1: enumerate manifest ==="
python3 enumerate.py \
    --out "$DEST/manifest.jsonl" \
    --year-start 1958 \
    --year-end 2026

echo
echo "=== Phase 2: download image bytes ==="
# Late-manifest records (Hubble/JWST/Landsat originals) average 60-80 MB each.
# With flush-every=500 + flush-bytes=8G the polars serialize-doubling briefly
# hit ~64 GB and OOM'd the node. Tighten to flush-every=100 + flush-bytes=2G
# so peak memory during flush stays <=5 GB. Resume is idempotent: load_done_ids
# skips rows already in existing shards.
python3 download.py \
    --manifest "$DEST/manifest.jsonl" \
    --out-dir  "$DEST/shards" \
    --concurrency 32 \
    --rps 30 \
    --flush-every 100 \
    --flush-bytes $((2*1024*1024*1024))

echo
echo "=== Final summary ==="
du -sh "$DEST"
ls -la "$DEST/shards/" | tail -20
