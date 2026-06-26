#!/bin/bash
# Download Visual Genome v1.4 — region descriptions, objects, relationships,
# plus image bytes (~108K images, ~14 GB).  CC-BY-4.0, ground-truth grounding
# annotations from human annotators (Ranjay Krishna et al., 2017).
#
# HF mirror (ranjaykrishna/visual_genome) ships only a loader script; data
# is hosted at Stanford. We fetch the canonical zips directly with wget.
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-visualgenome
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/visualgenome-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/visualgenome-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=SD-69241-apertus-1-5

set -euo pipefail

DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/stage2/visual_genome"
ANN_DIR="${DEST_DIR}/annotations"
IMG_DIR="${DEST_DIR}/images"

mkdir -p "${ANN_DIR}" "${IMG_DIR}"

echo "========================================"
echo "Visual Genome v1.4 download"
echo "Dest annotations: ${ANN_DIR}"
echo "Dest images:      ${IMG_DIR}"
echo "========================================"

# Annotation URLs — Stanford v1.4 canonical
ANN_BASE="https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset"
ANN_FILES=(
    "image_data.json.zip"            # ~6 MB  — image_id → url, w, h
    "region_descriptions.json.zip"   # ~480 MB — 5.4M regions with bboxes + phrases
    "objects.json.zip"               # ~270 MB — 3.8M objects with bboxes + names
    "relationships.json.zip"         # ~280 MB — 2.3M (subj, predicate, obj) triples
    "attributes.json.zip"            # ~190 MB — 2.8M object attributes
    "region_graphs.json.zip"         # ~390 MB — region scene graphs
    "question_answers.json.zip"      # ~190 MB — 1.7M VQA pairs
)

# Image zips — Stanford rak248 endpoint
IMG_PART1="https://cs.stanford.edu/people/rak248/VG_100K/images.zip"   # ~9 GB, ~62K images
IMG_PART2="https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip" # ~5 GB, ~46K images

# wget defaults: retry transient failures, continue partial files, parallel where possible
WGET_OPTS=(
    --tries=10
    --timeout=120
    --waitretry=5
    --retry-connrefused
    --continue
    --no-verbose
    --show-progress
    --progress=dot:giga
)

echo
echo "[$(date '+%F %T')] === downloading annotations ==="
for f in "${ANN_FILES[@]}"; do
    url="${ANN_BASE}/${f}"
    out="${ANN_DIR}/${f}"
    if [ -s "${out%.zip}" ] || [ -s "${out%.zip}.json" ]; then
        echo "  [skip] ${f} (already extracted)"
        continue
    fi
    echo "  → ${url}"
    wget "${WGET_OPTS[@]}" -O "${out}" "${url}"
done

echo
echo "[$(date '+%F %T')] === downloading image zips ==="
for url in "${IMG_PART1}" "${IMG_PART2}"; do
    fname=$(basename "${url}")
    out="${IMG_DIR}/${fname}"
    if [ -s "${out}" ] || [ -d "${out%.zip}" ]; then
        echo "  [skip] ${fname}"
        continue
    fi
    echo "  → ${url}"
    wget "${WGET_OPTS[@]}" -O "${out}" "${url}"
done

echo
echo "[$(date '+%F %T')] === extracting annotations ==="
cd "${ANN_DIR}"
for f in *.zip; do
    [ -e "${f%.zip}" ] || [ -e "${f%.zip}.json" ] && { echo "  [skip] ${f}"; continue; }
    echo "  unzip ${f}"
    unzip -o -q "${f}"
done

echo
echo "[$(date '+%F %T')] === extracting image zips ==="
cd "${IMG_DIR}"
for f in *.zip; do
    if [ -d "${f%.zip}" ]; then
        echo "  [skip] ${f}"; continue
    fi
    echo "  unzip ${f} (this is slow, ~108K small files)"
    unzip -o -q "${f}"
done

echo
echo "[$(date '+%F %T')] === cleanup zip files ==="
rm -f "${ANN_DIR}"/*.zip
rm -f "${IMG_DIR}"/*.zip

echo
echo "[$(date '+%F %T')] === final inventory ==="
echo "annotations:"
ls -lh "${ANN_DIR}"
echo
echo "image dirs (counts):"
for d in "${IMG_DIR}"/*/; do
    [ -d "${d}" ] || continue
    n=$(find "${d}" -maxdepth 1 -name '*.jpg' | wc -l)
    echo "  ${d}: ${n} images"
done
echo
echo "total dataset size:"
du -sh "${DEST_DIR}"
echo "Done."
