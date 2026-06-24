#!/bin/bash
#SBATCH --job-name=smithsonian-images
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_26_02
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=288
#SBATCH --array=0-20
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/smithsonian-images-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/smithsonian-images-%A_%a.err

# Smithsonian Open Access — array job, one collection per task, parallel.
# aws s3 sync is idempotent, so re-running a partially-downloaded collection
# just resumes. Excludes TIFF masters (same visual content as JPEG at 2-4x
# bytes, no training value).

set -euo pipefail

# Keep this list in the same order as `aws s3 ls s3://smithsonian-open-access/media/`.
# 21 collections => --array=0-20.
COLLECTIONS=(aaa acm cfch chsdm fm fs hmsg nasm nmaahc nmafa nmah nmai nmnh npg npm nzp ocio saam sf sg sia)

COL="${COLLECTIONS[$SLURM_ARRAY_TASK_ID]}"
DEST=/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/s3___smithsonian___OpenAccess/media/$COL

mkdir -p "$DEST"

export PATH="/users/xyixuan/.local/aarch64/bin:$PATH"

# Per-task AWS config file to avoid concurrent writes corrupting
# ~/.aws/config when many array tasks start simultaneously.
export AWS_CONFIG_FILE="$(mktemp -t aws-$COL-XXXX.cfg)"
cat > "$AWS_CONFIG_FILE" << 'EOF'
[default]
s3 =
    max_concurrent_requests = 128
    max_queue_size = 4000
    multipart_threshold = 64MB
EOF
trap 'rm -f "$AWS_CONFIG_FILE"' EXIT

aws --version

echo "=== [$COL] start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
time aws s3 sync --no-sign-request \
    "s3://smithsonian-open-access/media/$COL/" \
    "$DEST/" \
    --exclude "*.tif" --exclude "*.tiff" \
    --cli-read-timeout 0 \
    --cli-connect-timeout 60 \
    --only-show-errors

sz=$(du -sh "$DEST" 2>/dev/null | cut -f1 || echo "?")
n=$(find "$DEST" -type f 2>/dev/null | wc -l)
echo "=== [$COL] done $(date -u +%Y-%m-%dT%H:%M:%SZ): $n files, $sz ==="
