#!/bin/bash
# Download Suno MP3s from CDN using aria2c
# Requires urls.txt from download_metadata.slurm
#SBATCH --account=infra01
#SBATCH --job-name=suno-audio
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/suno-audio-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/suno-audio-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tools

DEST_DIR=${DEST_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/suno}
CLIPS_DIR="${DEST_DIR}/clips"
URLS_FILE="${DEST_DIR}/urls.txt"

if [ ! -f "${URLS_FILE}" ]; then
    echo "ERROR: ${URLS_FILE} not found. Run download_metadata.slurm first."
    exit 1
fi

mkdir -p "${CLIPS_DIR}"

# Build a filtered URL list: skip already-downloaded clips
REMAINING_FILE="${DEST_DIR}/urls_remaining.txt"
echo "[$(date '+%F %T')] Filtering already-downloaded clips..."
python3 -c "
import os
clips = set(os.listdir('${CLIPS_DIR}'))
seen = set()
written = skipped = 0
with open('${URLS_FILE}') as fin, open('${REMAINING_FILE}', 'w') as fout:
    for line in fin:
        if line.startswith('  out='):
            fname = line.strip().split('=', 1)[1]
            if fname in clips or fname in seen:
                skipped += 1
                continue
            seen.add(fname)
            fout.write(url_line)
            fout.write(line)
            written += 1
        else:
            url_line = line
print(f'Remaining: {written:,} (skipped {skipped:,} downloaded/duplicate)')
"

echo "========================================"
echo "Suno Audio Download"
echo "URLs file: ${REMAINING_FILE}"
echo "Dest: ${CLIPS_DIR}"
echo "Remaining: $(grep -c '^http' "${REMAINING_FILE}")"
echo "========================================"

echo "[$(date '+%F %T')] Starting aria2c download..."
aria2c \
    -i "${REMAINING_FILE}" \
    -d "${CLIPS_DIR}" \
    -j 64 \
    -x 4 \
    --max-overall-download-limit=500M \
    --auto-file-renaming=false \
    --continue=true \
    --max-tries=5 \
    --retry-wait=5 \
    --timeout=30 \
    --log="${DEST_DIR}/aria2c.log" \
    --log-level=warn \
    --summary-interval=60

echo "[$(date '+%F %T')] Download finished"
echo "Files downloaded: $(ls "${CLIPS_DIR}" | wc -l)"
