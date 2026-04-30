#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=vietspeech-tok
#SBATCH --reservation=SD-69241-apertus-1-5-6
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/tokenize/vietspeech_tok_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/tokenize/vietspeech_tok_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

RESUME=${1:-false}
if [ "${RESUME}" = "resume" ]; then RESUME=true; fi

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
RUNTIME_ENV_SH="${REPO_DIR}/scripts/utils/source_lhotse_runtime.sh"
mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/tokenize
TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * 4))

echo "=========================================="
echo "VietSpeech Direct Tokenization (~1144h, ~5min on 4 GPUs)"
echo "Job ID: ${SLURM_JOB_ID}, Nodes: ${SLURM_JOB_NUM_NODES}, GPUs: ${TOTAL_GPUS}"
echo "=========================================="

srun --kill-on-bad-exit=0 -u bash -c '
    set -e
    export LD_PRELOAD=/opt/venv/lib/python3.12/site-packages/ray/core/libjemalloc.so
    export INSTALL_TORCHAUDIO=1
    export INSTALL_ONCE_PER_NODE=1
    source '"${RUNTIME_ENV_SH}"'
    cd '"${REPO_DIR}"'
    python -m audio_tokenization.tokenize \
        dataset=audio_text/vietspeech_direct \
        mode=audio_text \
        audio_text_format=direct \
        num_gpus='"${TOTAL_GPUS}"' \
        resume='"${RESUME}"' \
        wandb.name="vietspeech_tok_'"${SLURM_JOB_NUM_NODES}"'node" \
        wandb.tags="[tokenization,vietspeech,direct,cooldown]"
'

# Create symlink under cooldown directory
TOKENIZED_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/tokenized"
COOLDOWN_DIR="${TOKENIZED_DIR}/cooldown"
mkdir -p "${COOLDOWN_DIR}"
ln -sfn "${TOKENIZED_DIR}/transcribe/vietspeech" "${COOLDOWN_DIR}/asr_vietspeech"
echo "Symlinked: ${COOLDOWN_DIR}/asr_vietspeech -> ${TOKENIZED_DIR}/transcribe/vietspeech"
