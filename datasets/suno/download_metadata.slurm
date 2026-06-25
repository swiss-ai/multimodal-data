#!/bin/bash
# Download Suno metadata parquet from HuggingFace, then extract audio URLs
#SBATCH --account=infra01
#SBATCH --job-name=suno-meta
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/suno-meta-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/suno-meta-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DEST_DIR=${DEST_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/suno}
PARQUET_DIR="${DEST_DIR}/parquet"
URLS_FILE="${DEST_DIR}/urls.txt"

mkdir -p "${DEST_DIR}"

uv pip install --python /opt/venv/bin/python hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

echo "========================================"
echo "Suno Metadata Download"
echo "Dest: ${PARQUET_DIR}"
echo "========================================"

# Step 1: Download parquet from HF
echo "[$(date '+%F %T')] Downloading parquet from HF..."
huggingface-cli download nyuuzyou/suno \
    --repo-type dataset \
    --local-dir "${PARQUET_DIR}"

echo "[$(date '+%F %T')] Download done"

# Step 2: Extract audio URLs from parquet
echo "[$(date '+%F %T')] Extracting audio URLs..."
python3 -c "
import pandas as pd, glob

files = sorted(glob.glob('${PARQUET_DIR}/**/*.parquet', recursive=True))
print(f'Found {len(files)} parquet files')

written = 0
skipped = 0
seen = set()
with open('${URLS_FILE}', 'w') as f:
    for pf in files:
        df = pd.read_parquet(pf, columns=['id', 'audio_url'])
        for _, row in df.iterrows():
            url, sid = row.get('audio_url'), row.get('id')
            if not url or not sid or pd.isna(url) or pd.isna(sid):
                skipped += 1
                continue
            if sid in seen:
                skipped += 1
                continue
            seen.add(sid)
            f.write(f'{url}\n')
            f.write(f'  out={sid}.mp3\n')
            written += 1

print(f'Wrote {written:,} unique URLs (skipped {skipped:,} null/duplicate)')
"

echo "[$(date '+%F %T')] URLs written to ${URLS_FILE}"
echo "Total URLs: $(grep -c '^http' "${URLS_FILE}")"

# Cleanup HF cache
rm -rf "${HF_HUB_CACHE}"
echo "Done."
