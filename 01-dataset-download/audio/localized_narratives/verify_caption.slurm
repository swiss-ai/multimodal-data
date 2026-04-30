#!/bin/bash
# Run verify_caption.py on a GPU node inside the NeMo container.
#
# Usage:
#   sbatch verify_caption.slurm
#   LN_GROUPS="coco_val,ade20k_validation,flickr30k_test" N_PER=8 \
#       sbatch verify_caption.slurm
#
# Note: do NOT use the bash variable name `GROUPS` — bash defines it as a
# readonly builtin (supplementary group IDs) and will silently override our
# default with a numeric value like "65603".
#
#SBATCH --account=infra01
#SBATCH --job-name=ln-verify
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/localized_narratives/logs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/localized_narratives/logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal

set -euo pipefail

LN_GROUPS="${LN_GROUPS:-coco_val,ade20k_validation,flickr30k_test}"
N_PER="${N_PER:-8}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"

SCRIPT_DIR="/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/localized_narratives"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
OUTPUT_JSON="${SCRIPT_DIR}/logs/verify_caption-${SLURM_JOB_ID}.json"

mkdir -p "${SCRIPT_DIR}/logs"

echo "=========================================="
echo "LN caption verification (Parakeet ASR)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)"
echo "Groups:    ${LN_GROUPS}"
echo "N/group:   ${N_PER}"
echo "Seed:      ${SEED}"
echo "Output:    ${OUTPUT_JSON}"
echo "=========================================="

# nemo_26_02 (nv25.11, aarch64) ships torchaudio .so files with an ABI
# mismatch against its custom torch build. Importing torchaudio crashes —
# but renaming the .so files makes torchaudio fall back to pure-Python mode
# (Resample etc. still work). Container overlay is per-job, so re-apply each
# run. Idempotent: skips if already renamed.
TA_LIB="/usr/local/lib/python3.12/dist-packages/torchaudio/lib"
if [[ -d "${TA_LIB}" ]]; then
    for so in _torchaudio.abi3.so libtorchaudio.abi3.so torchaudio_prefixctc.abi3.so; do
        if [[ -f "${TA_LIB}/${so}" ]]; then
            mv "${TA_LIB}/${so}" "${TA_LIB}/${so}.bak"
            echo "  patched torchaudio: ${so} → ${so}.bak"
        fi
    done
fi


"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_caption.py" \
    --groups "${LN_GROUPS}" \
    --n-per-group "${N_PER}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --device cuda \
    --output-json "${OUTPUT_JSON}"
