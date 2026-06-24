#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ytc-asr-tok
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/tokenize/granary_ytc_asr_tok_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/tokenize/granary_ytc_asr_tok_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

RESUME=${1:-false}
if [ "${RESUME}" = "resume" ]; then
    RESUME=true
fi

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
RUNTIME_ENV_SH="${REPO_DIR}/scripts/utils/source_lhotse_runtime.sh"
mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/tokenize

TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * 4))

# YTC clips heavily overlap in time (median gap = -9.4s), making interleaving
# unsuitable. Direct mode treats each clip independently.

echo "=========================================="
echo "YTC ASR Direct Tokenization"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Total GPUs: ${TOTAL_GPUS}"
echo "Resume: ${RESUME}"
echo "=========================================="

srun --kill-on-bad-exit=0 -u bash -c '
    set -e
    export LD_PRELOAD=/opt/venv/lib/python3.12/site-packages/ray/core/libjemalloc.so
    export INSTALL_TORCHAUDIO=1
    export INSTALL_ONCE_PER_NODE=1
    source '"${RUNTIME_ENV_SH}"'

    cd '"${REPO_DIR}"'

    python -m audio_tokenization.tokenize \
        dataset=audio_text/granary_ytc_asr_direct \
        mode=audio_text \
        audio_text_format=direct \
        num_gpus='"${TOTAL_GPUS}"' \
        resume='"${RESUME}"' \
        wandb.name="granary_ytc_asr_tok_'"${SLURM_JOB_NUM_NODES}"'nodes" \
        wandb.tags="[tokenization,granary_ytc,asr,direct,'"${SLURM_JOB_NUM_NODES}"'nodes]"
'

echo "=========================================="
echo "YTC ASR tokenization finished"
echo "=========================================="
