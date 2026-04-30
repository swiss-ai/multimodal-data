#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=tok-ups
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/tok_ups_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/tok_ups_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

# Usage:
#   sbatch tokenize.slurm          # fresh run
#   sbatch tokenize.slurm resume   # resume from checkpoints
RESUME=${1:-false}

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
RUNTIME_ENV_SH="${REPO_DIR}/scripts/utils/source_lhotse_runtime.sh"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * 4))

echo "=========================================="
echo "UPS Tokenization (Lhotse DDP)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Total GPUs: ${TOTAL_GPUS}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Resume: ${RESUME}"
echo "=========================================="

srun --kill-on-bad-exit=0 -u bash -c '
    set -e
    export LD_PRELOAD=/opt/venv/lib/python3.12/site-packages/ray/core/libjemalloc.so
    export REPO_DIR='"${REPO_DIR}"'
    export INSTALL_TORCHAUDIO=1
    export INSTALL_ONCE_PER_NODE=1
    export RUNTIME_READY_FILE=/tmp/.lhotse_runtime_ready_${SLURM_JOB_ID}
    export PRINT_LHOTSE_RUNTIME=0
    source '"${RUNTIME_ENV_SH}"'

    cd '"${REPO_DIR}"'

    python -m audio_tokenization.tokenize \
        dataset=unsupervised_peoples_speech_lhotse \
        num_gpus='"${TOTAL_GPUS}"' \
        resume='"${RESUME}"' \
        wandb.name="ups_v2_tok_'"${SLURM_JOB_NUM_NODES}"'nodes" \
        wandb.tags="[tokenization,ups_v2,'"${SLURM_JOB_NUM_NODES}"'nodes]"
'

echo "=========================================="
echo "UPS tokenization finished"
echo "=========================================="
