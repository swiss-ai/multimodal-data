#!/bin/bash
# Pack Suno clips into tar shards, split by prompt availability.
# Creates shards_s2/ (with prompt) and shards_s1/ (without prompt).
# Each tar contains ~1000 clips with .mp3 + .url + optional .txt sidecars.
# Exports per-split metadata JSONL, then deletes original clips and sidecars.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SDIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/suno
CLIPS_DIR=${SDIR}/clips
PARQUET=${SDIR}/parquet/data/train-00000-of-00001.parquet

mkdir -p ${SDIR}/shards_s1 ${SDIR}/shards_s2

# Step 1: Split clips, create sidecars, export metadata
echo "[$(date '+%F %T')] Splitting clips by prompt and creating sidecars..."
python3 ${SCRIPT_DIR}/split_by_prompt.py \
    --parquet ${PARQUET} \
    --clips-dir ${CLIPS_DIR} \
    --output-dir ${SDIR}

# Step 2: Split file lists into manifests
# with_prompt: ~3 files per clip (mp3 + url + txt) → 3000 lines = ~1000 clips
# without_prompt: ~2 files per clip (mp3 + url) → 2000 lines = ~1000 clips
split -l 3000 -d -a 4 ${SDIR}/files_with_prompt.txt ${SDIR}/manifest_wp_
split -l 2000 -d -a 4 ${SDIR}/files_without_prompt.txt ${SDIR}/manifest_np_

# Step 3: Pack both sets in parallel (16 tar processes per set)
echo "[$(date '+%F %T')] Packing tars in parallel..."
(
for f in ${SDIR}/manifest_wp_*; do
    idx=$(basename $f | sed 's/manifest_wp_//')
    tar cf ${SDIR}/shards_s2/part-${idx}.tar -T $f &
    if (( $(jobs -r | wc -l) >= 16 )); then wait -n; fi
done
wait
echo "shards_s2 done: $(ls ${SDIR}/shards_s2/part-*.tar | wc -l) tars"
) &

(
for f in ${SDIR}/manifest_np_*; do
    idx=$(basename $f | sed 's/manifest_np_//')
    tar cf ${SDIR}/shards_s1/part-${idx}.tar -T $f &
    if (( $(jobs -r | wc -l) >= 16 )); then wait -n; fi
done
wait
echo "shards_s1 done: $(ls ${SDIR}/shards_s1/part-*.tar | wc -l) tars"
) &

wait

# Step 4: Move metadata into shard dirs
mv ${SDIR}/metadata_with_prompt.jsonl ${SDIR}/shards_s2/metadata.jsonl
mv ${SDIR}/metadata_without_prompt.jsonl ${SDIR}/shards_s1/metadata.jsonl

# Step 5: Delete original clips and sidecars
echo "[$(date '+%F %T')] Deleting original clips and sidecars..."
cat ${SDIR}/files_with_prompt.txt ${SDIR}/files_without_prompt.txt | xargs -P 16 rm -f
echo "[$(date '+%F %T')] Remaining in clips dir: $(ls ${CLIPS_DIR} | wc -l)"

# Cleanup manifests and file lists
rm -f ${SDIR}/manifest_wp_* ${SDIR}/manifest_np_*
rm -f ${SDIR}/files_with_prompt.txt ${SDIR}/files_without_prompt.txt

echo "[$(date '+%F %T')] All done"
