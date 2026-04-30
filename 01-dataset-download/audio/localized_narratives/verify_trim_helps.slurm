#!/bin/bash
# Does trimming with timed_caption boundaries reduce Parakeet WER?
#
#SBATCH --account=infra01
#SBATCH --job-name=ln-trim
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/localized_narratives/logs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/localized_narratives/logs/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal

set -euo pipefail

SCRIPT_DIR="/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/localized_narratives"
mkdir -p "${SCRIPT_DIR}/logs"

# Idempotent torchaudio ABI fix (no-op if already patched)
TA_LIB="/usr/local/lib/python3.12/dist-packages/torchaudio/lib"
if [[ -d "${TA_LIB}" ]]; then
    for so in _torchaudio.abi3.so libtorchaudio.abi3.so torchaudio_prefixctc.abi3.so; do
        [[ -f "${TA_LIB}/${so}" ]] && mv "${TA_LIB}/${so}" "${TA_LIB}/${so}.bak"
    done
fi

/opt/venv/bin/python "${SCRIPT_DIR}/verify_trim_helps.py" \
    --n-per-group "${N_PER:-8}" \
    --seed "${SEED:-42}" \
    --pad-ms "${PAD_MS:-100}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --device cuda
