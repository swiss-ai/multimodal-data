#!/bin/bash
# Download OS-Copilot/OS-Atlas-data (~760 GiB, Apache-2.0).
# GUI grounding pretraining data for OS-ATLAS, 3 domains:
#   web_domain/    (557 GiB) — fineweb 10 shards + seeclick split + seeclick_web.json
#   mobile_domain/ (95 GiB)  — amex split + rico/mobile/UIBert zips + JSONs
#   desktop_domain/(75 GiB)  — windows split + macos/linux zips + JSONs
#   IG_data/       (33 GiB)  — amex + aitz + mind2web + guiact zips
# Bbox format in JSON: [left, top, right, bottom] as 0–1 decimals (image-relative).
#
# Post-download: 3 sets of files are SPLIT and need cat-concatenation before extraction:
#   mobile_domain/amex_images_part_{aa,ab,ac}    -> amex_images.zip
#   desktop_domain/windows_image_{aa,ab,ac,ad}   -> windows_image.zip
#   web_domain/seeclick_web_image_{aa,ab,ac,ad}  -> seeclick_web_image.zip
# This script does NOT auto-concatenate (would double disk footprint to ~1.5 TiB).
#
# Usage:
#   sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/os_atlas_data/download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-os-atlas-data
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/os-atlas-data-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/os-atlas-data-%x-%A.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_REPO="OS-Copilot/OS-Atlas-data"
REVISION="e3a4c90c5f6129c25efdaa0671b08a11f7cb8f3f"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___OS-Copilot___OS-Atlas-data"
SCRIPT_PATH="01-dataset-download/image/os_atlas_data/download.slurm"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

mkdir -p "${DEST_DIR}"

echo "========================================"
echo "OS-Atlas-data Download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "Expected: ~760 GiB, Apache-2.0, 42 files (19 zips + 11 jsons + 12 split parts)"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${DEST_DIR}" \
    --max-workers 64

echo "[$(date '+%F %T')] Download finished"

cat > "${DEST_DIR}/_SOURCE.json" <<EOF
{
  "dataset_id": "hf___OS-Copilot___OS-Atlas-data",
  "repo_url": "https://huggingface.co/datasets/${DATASET_REPO}",
  "revision": "${REVISION}",
  "license": "apache-2.0",
  "data_type": "pretraining_grounding",
  "task": "gui_grounding_pretraining",
  "bbox_format": "[left, top, right, bottom] as 0-1 decimal of image dim",
  "download_method": "huggingface-cli download with HF_HUB_ENABLE_HF_TRANSFER=1",
  "downloaded_at_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "script": "${SCRIPT_PATH}",
  "size_gib": 760,
  "notes": "GUI grounding pretraining data for OS-ATLAS (arxiv:2410.23218). 3 domains: mobile/desktop/web + IG_data. Bbox is [l,t,r,b] in 0-1 decimal. 3 sets of split files need cat-concatenation before extraction: mobile_domain/amex_images_part_{aa..ac} -> amex_images.zip; desktop_domain/windows_image_{aa..ad} -> windows_image.zip; web_domain/seeclick_web_image_{aa..ad} -> seeclick_web_image.zip. fineweb_3m_s** zips are individual standalone shards (no concat). Companion JSONs hold the actual labels."
}
EOF

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "[$(date '+%F %T')] Done"
