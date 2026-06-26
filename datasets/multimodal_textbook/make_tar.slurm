#!/bin/bash
#SBATCH --job-name=damo-textbook-tar
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_26_02
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=288
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/damo-textbook-tar-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/damo-textbook-tar-%j.err

# Bundle the DAMO Multimodal Textbook tree into a single .tar (no compression
# — JPEGs are already compressed, gzip gives <1% at 10× the CPU).
#
# Why: 6.5M keyframe JPEGs create enormous inode pressure on Capstor.
# A single .tar bundle is one inode. mksquashfs was too slow for the 12h
# SLURM walltime (extrapolated ~23h on capstor-to-capstor), so we use tar
# instead. Training-time loading will need tarfile-streaming (webdataset /
# energon) rather than squashfuse mount.

set -euo pipefail

SRC_PARENT=/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown
SRC_NAME=damo_multimodal_textbook
DST=/capstor/store/cscs/swissai/infra01/vision-datasets/raw/archive/${SRC_NAME}.tar

mkdir -p "$(dirname "$DST")"

echo "=== input ==="
ls -la "$SRC_PARENT/$SRC_NAME" | head
df -h "$(dirname "$DST")" | head
echo
echo "=== tar start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# -cf      create file (no compression)
# --sort=name  deterministic ordering (makes incremental diffs / resume tractable)
# -C parent dir, so the paths inside the tar are relative to damo_multimodal_textbook/
time tar -cf "$DST" \
    --sort=name \
    -C "$SRC_PARENT" \
    "$SRC_NAME"

echo
echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ls -lh "$DST"
echo
echo "file count inside tar (first check):"
tar -tf "$DST" | wc -l
echo
echo "Sanity check: first 10 entries:"
tar -tf "$DST" | head -10

echo
echo "Training-time loader hint (webdataset-style):"
echo "  wds = webdataset.WebDataset('$DST')"
echo "or stream with python tarfile module."
