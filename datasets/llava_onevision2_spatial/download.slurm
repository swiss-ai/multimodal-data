#!/bin/bash
# Download mvp-lab/LLaVA-OneVision-2-Data — ONLY spatial/ + spatial_instruct/ (~255 GiB, Apache-2.0).
# Skips the ~62 TB video/image bulk of the full repo via --include filter.
#   spatial/         251 GiB  — 84 WebDataset .tar shards (image media + depth maps)
#   spatial_instruct/  4.4 GiB — 36 JSONL instruction files (refcoco, visual_genome, pointing,
#                                ca1m multi-view, osd, crosspoint, erqa, roborefer_sim)
# Image-only (no video). Some 3D subsets carry RGB-D depth maps; depth paths may reference
# an external RoboBrain2.5 source not included here.
#
# Usage:
#   sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/llava_onevision2_spatial/download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-lov2-spatial
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/lov2-spatial-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/lov2-spatial-%x-%A.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_REPO="mvp-lab/LLaVA-OneVision-2-Data"
REVISION="e73747a5aff28d10c6207841f95e290b5467ca07"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___mvp-lab___LLaVA-OneVision-2-Data"
SCRIPT_PATH="01-dataset-download/image/llava_onevision2_spatial/download.slurm"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

mkdir -p "${DEST_DIR}"

echo "========================================"
echo "LLaVA-OneVision-2-Data: spatial/ + spatial_instruct/ ONLY"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "Expected: ~255 GiB (84 spatial tars + 36 spatial_instruct jsonl); video bulk excluded"
echo "========================================"

# Single --include flag, multiple patterns (multiple --include flags would silently overwrite).
huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${DEST_DIR}" \
    --include "spatial/*" "spatial_instruct/*" "README.md" \
    --max-workers 64

echo "[$(date '+%F %T')] Download finished"

cat > "${DEST_DIR}/_SOURCE.json" <<EOF
{
  "dataset_id": "hf___mvp-lab___LLaVA-OneVision-2-Data",
  "repo_url": "https://huggingface.co/datasets/${DATASET_REPO}",
  "revision": "${REVISION}",
  "license": "apache-2.0",
  "subset": "spatial/ + spatial_instruct/ only (video bulk excluded)",
  "data_type": "sft",
  "task": "spatial_reasoning_grounding",
  "modality": "image (RGB + optional depth); no video",
  "download_method": "huggingface-cli download with --include spatial/* spatial_instruct/*, HF_HUB_ENABLE_HF_TRANSFER=1",
  "downloaded_at_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "script": "${SCRIPT_PATH}",
  "size_gib": 255,
  "notes": "Only the spatial slice of LLaVA-OneVision-2-Data. spatial/ = 84 WebDataset tar shards (image media, ~251 GiB). spatial_instruct/ = 36 JSONL (~4.4 GiB). Image-only (no video). Full repo is ~62 TB incl. video (mid_training_video/) NOT downloaded.",
  "per_subset_license_audit": {
    "osd_* (OpenImages 2D part of RefSpatial)": "apache-2.0 — COMMERCIAL OK",
    "roborefer_sim (Blender Simulator part of RefSpatial)": "apache-2.0 — COMMERCIAL OK",
    "visual_genome": "cc-by-4.0 — but ALREADY HAVE in cooldown mix (raw/stage2/visual_genome); skip",
    "refcoco/refcoco+/refcocog": "MS-COCO image provenance — research-safe, commercial gray",
    "erqa / crosspoint / pointing": "unverified, small (~15-80 MiB each)",
    "ca1m_* (Apple Cubify Anything, the 3D part)": "CC-BY-NC-ND — NON-COMMERCIAL + NO-DERIVATIVES. DO NOT train on for any commercial Apertus checkpoint. Filter ca1m_* records at conversion time."
  },
  "commercial_clean_subsets": ["osd_choice", "osd_reasoning", "osd_template", "roborefer_sim"],
  "depth_note": "spatial_instruct records carry a 'depth' field pointing to /ov2/.../RoboBrain2.5-Spatial-Part-Dataset/...png — external path NOT in this repo. RGB images ARE in spatial/ tarballs; depth maps are not. Canonical RGB-D source is JingkunAn/RefSpatial if depth is needed."
}
EOF

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "[$(date '+%F %T')] Done"
