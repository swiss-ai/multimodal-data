#!/bin/bash
# Download 2077AIDataFoundation/ChartNet_RealWorldChart (~2.8 GiB, CC-BY-4.0).
# ~30K real-world charts with detailed captions. Companion to ibm-granite/ChartNet.
#
# Usage:
#   sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/chartnet_realworldchart/download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-chartnet-rwc
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/chartnet-rwc-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/chartnet-rwc-%x-%A.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=normal

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_REPO="2077AIDataFoundation/ChartNet_RealWorldChart"
REVISION="42a9a44b222196b333f983505d77e5f3debddf4a"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___2077AIDataFoundation___ChartNet_RealWorldChart"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"
mkdir -p "${DEST_DIR}"

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${DEST_DIR}" \
    --max-workers 32

cat > "${DEST_DIR}/_SOURCE.json" <<EOF
{
  "dataset_id": "hf___2077AIDataFoundation___ChartNet_RealWorldChart",
  "repo_url": "https://huggingface.co/datasets/${DATASET_REPO}",
  "revision": "${REVISION}",
  "license": "cc-by-4.0",
  "data_type": "sft",
  "task": "chart_captioning_vqa",
  "download_method": "huggingface-cli download with HF_HUB_ENABLE_HF_TRANSFER=1",
  "downloaded_at_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "script": "01-dataset-download/image/chartnet_realworldchart/download.slurm",
  "size_gib": 2.8,
  "notes": "~30K real-world charts with detailed captions. Companion to ibm-granite/ChartNet core_permissive. Released by Abaka AI / 2077AI."
}
EOF

rm -rf "${HF_HUB_CACHE}"
echo "Done: $(du -sh ${DEST_DIR})"
