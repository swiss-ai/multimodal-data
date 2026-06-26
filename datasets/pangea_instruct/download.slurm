#!/bin/bash
# Download neulab/PangeaInstruct (~730 GiB, Apache 2.0, 39 languages).
# Multilingual VQA + QA SFT mix covering 8 categories:
#   caption / cultural / doc+chart / general / ocr / text-only / translation / LLaVA-Pretrain
# doc+chart subdir includes ChartQA, doc-vqa, table-vqa, Viet-Doc-VQA / Viet-OCR-VQA.
# Format: per-subset {data.json + images.tar|zip|tar.gz}, plus a top-level PangeaIns.json.
#
# Usage:
#   sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/pangea_instruct/download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-pangea-instruct
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/pangea-instruct-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/pangea-instruct-%x-%A.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_REPO="neulab/PangeaInstruct"
REVISION="d0819917abe1cae38c008de0ca172f885f1f26a4"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___neulab___PangeaInstruct"
SCRIPT_PATH="01-dataset-download/image/pangea_instruct/download.slurm"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

mkdir -p "${DEST_DIR}"

echo "========================================"
echo "PangeaInstruct Download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "Expected: ~730 GiB, Apache 2.0, 39 languages"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${DEST_DIR}" \
    --max-workers 64

echo "[$(date '+%F %T')] Download finished"

cat > "${DEST_DIR}/_SOURCE.json" <<EOF
{
  "dataset_id": "hf___neulab___PangeaInstruct",
  "repo_url": "https://huggingface.co/datasets/${DATASET_REPO}",
  "revision": "${REVISION}",
  "license": "apache-2.0",
  "download_method": "huggingface-cli download with HF_HUB_ENABLE_HF_TRANSFER=1",
  "downloaded_at_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "script": "${SCRIPT_PATH}",
  "notes": "Multilingual VQA+QA SFT mix (39 languages). Subdirs: caption/, cultural/, doc+chart/ (incl. table-vqa, ChartQA, doc-vqa, Viet-Doc-VQA, Viet-OCR-VQA), general/, ocr/, text-only/, translation/, LLaVA-Pretrain/. Master annotation file: PangeaIns.json. Image payloads ship as per-subset tar/zip/tar.gz that must be extracted before training."
}
EOF

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "[$(date '+%F %T')] Done"
