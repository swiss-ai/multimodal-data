#!/bin/bash
# Combine Visual Genome JSON annotations + JPEG bytes → sharded parquet
# (zstd compressed). Memory-heavy: loads ~5 GB of JSON in RAM during build.
#
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --job-name=vg-build-parquet
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/envs/nemo_25_11.toml
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/vg-parquet-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/vg-parquet-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive

set -euo pipefail

cd /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/visual_genome

echo "========================================"
echo "Visual Genome → parquet conversion"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname -s)"
echo "========================================"

srun -u --kill-on-bad-exit=0 bash -c "
  set -e
  python3 build_visual_genome_parquet.py
"

echo
echo "=== final output inventory ==="
DEST=/capstor/store/cscs/swissai/infra01/vision-datasets/raw/stage2/visual_genome/parquet
ls "$DEST" | head -5
echo "..."
NSHARDS=$(ls "$DEST"/*.parquet 2>/dev/null | wc -l)
echo "  parquet shards: $NSHARDS"
du -sh "$DEST"
echo "Done."
