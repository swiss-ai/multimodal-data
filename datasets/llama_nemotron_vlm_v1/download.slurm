#!/bin/bash
#SBATCH --job-name=nemotron-vlm-v1
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_26_02
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=288
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/nemotron-vlm-v1-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/nemotron-vlm-v1-%j.err

# NVIDIA Llama-Nemotron-VLM-Dataset-v1
#   https://huggingface.co/datasets/nvidia/Llama-Nemotron-VLM-Dataset-v1
#
# Strategy: huggingface-cli download with HF_TRANSFER.  The repo has mostly
# large JSONL + tarred image shards — few-but-large files → HF_TRANSFER wins.
#
# Under a "PD or permissive, no share-alike" licence rule we drop the 4
# CC-BY-SA sub-datasets:
#     ocr_4  (32.6 GB, rendered Wikipedia text)
#     ocr_5  (32.4 GB, rendered Wikipedia text)
#     ocr_7  ( 2.5 GB, TabRecSet tables)
#     vqa_9  (10.9 GB, textbook VQA)
# The remaining 17 sub-datasets are CC-BY-4.0 (ocr_6 has an additional
# DocLayNet attribution requirement — read its .md file before training).

set -euo pipefail

DEST=/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/nvidia_nemotron_vlm_v1
mkdir -p "$DEST"

# hf_transfer lives in the shared pip-packages dir (matches fine-t2i, etc.).
export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=600

echo "=== downloading nvidia/Llama-Nemotron-VLM-Dataset-v1 to $DEST ==="
echo "=== excluded (CC-BY-SA): ocr_4 ocr_5 ocr_7 vqa_9 ==="
date -u +%Y-%m-%dT%H:%M:%SZ

time huggingface-cli download nvidia/Llama-Nemotron-VLM-Dataset-v1 \
    --repo-type dataset \
    --local-dir "$DEST" \
    --exclude \
        "ocr_4.jsonl" "ocr_4.jsonl.idx" "ocr_4.md" "ocr_4_images/*" \
        "ocr_5.jsonl" "ocr_5.jsonl.idx" "ocr_5.md" "ocr_5_images/*" \
        "ocr_7.jsonl" "ocr_7.jsonl.idx" "ocr_7.md" \
        "vqa_9.jsonl" "vqa_9.jsonl.idx" "vqa_9.md" "vqa_9_images/*"

echo
echo "=== final summary ==="
du -sh "$DEST"
find "$DEST" -maxdepth 1 -name "*.jsonl" | sort
find "$DEST" -maxdepth 1 -type d -name "*_images" | sort
date -u +%Y-%m-%dT%H:%M:%SZ
