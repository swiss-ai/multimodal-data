#!/bin/bash
# Download CorentinDumery/MixCount (~24 GiB, CC-BY-4.0, open-vocabulary object counting).
# Raw supervised counting/detection data — NOT instruction format. Need prompt-templating
# at prep time to use as SFT (see fields: total_count, class_counts, class_descriptions, objects.bbox).
# 61 parquet shards under parquet_data/, train + val + test splits.
# Paper: arXiv 2605.18063 (Dumery et al., EPFL / Oxford / Northwestern).
#
# Usage:
#   sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/mixcount/download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-mixcount
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mixcount-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mixcount-%x-%A.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_REPO="CorentinDumery/MixCount"
REVISION="d7b8243c842cbfecce4f882d209828de56835228"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___CorentinDumery___MixCount"
SCRIPT_PATH="01-dataset-download/image/mixcount/download.slurm"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

mkdir -p "${DEST_DIR}"

echo "========================================"
echo "MixCount Download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "Expected: ~24 GiB, CC-BY-4.0, 61 parquet shards (train+val+test)"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${DEST_DIR}" \
    --max-workers 32

echo "[$(date '+%F %T')] Download finished"

cat > "${DEST_DIR}/_SOURCE.json" <<EOF
{
  "dataset_id": "hf___CorentinDumery___MixCount",
  "repo_url": "https://huggingface.co/datasets/${DATASET_REPO}",
  "revision": "${REVISION}",
  "license": "cc-by-4.0",
  "data_type": "raw_supervised_counting_detection",
  "task": "open_vocabulary_object_counting",
  "bbox_format": "[x1, y1, x2, y2] float (verify normalization from row sample)",
  "download_method": "huggingface-cli download with HF_HUB_ENABLE_HF_TRANSFER=1",
  "downloaded_at_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "script": "${SCRIPT_PATH}",
  "size_gib": 24,
  "paper": "arxiv:2605.18063",
  "splits": ["train", "val", "test"],
  "fields": ["image", "file_name", "total_count", "num_classes", "class_names", "class_counts", "class_descriptions", "objects.bbox", "objects.category"],
  "notes": "Open-vocabulary object counting (Dumery et al., EPFL/Oxford/NW). NOT shipped as conversation format. Apply prompt templating at prep time to convert to ShareGPT JSONL: e.g. 'How many {class_name} are in the image?' -> '{class_count}'. Companion to CountBench / FSC-147 / PixMo-Count."
}
EOF

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "[$(date '+%F %T')] Done"
