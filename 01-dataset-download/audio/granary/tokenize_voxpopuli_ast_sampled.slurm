#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=vp-ast-tok
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/tokenize/voxpopuli_ast_sampled_tok_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/tokenize/voxpopuli_ast_sampled_tok_%j.err
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
LOG_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/tokenize
mkdir -p "${LOG_DIR}"

TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * 4))

echo "=========================================="
echo "VoxPopuli AST Sampled Tokenization"
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
        dataset=audio_text/voxpopuli_ast_sampled_direct \
        mode=audio_text \
        audio_text_format=direct \
        audio_text_task=translate \
        num_gpus='"${TOTAL_GPUS}"' \
        resume='"${RESUME}"' \
        wandb.name="voxpopuli_ast_sampled_tok_'"${SLURM_JOB_NUM_NODES}"'nodes" \
        wandb.tags="[tokenization,voxpopuli,ast,sampled,direct,translate,'"${SLURM_JOB_NUM_NODES}"'nodes]"
'

echo "=========================================="
echo "VoxPopuli AST sampled tokenization finished"
echo "=========================================="
