#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=tok-aishell
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/aishell/logs/tok_aishell_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/aishell/logs/tok_aishell_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
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
mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/aishell/logs

TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * 4))

echo "=========================================="
echo "AISHELL-1+3 Direct Tokenization"
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
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:'"${REPO_DIR}"':${PYTHONPATH:-}

    if [ "${SLURM_LOCALID}" = "0" ]; then
        uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation \
            git+https://github.com/pytorch/audio.git@release/2.9
        touch /tmp/.torchaudio_ready_${SLURM_JOB_ID}
    else
        while [ ! -f /tmp/.torchaudio_ready_${SLURM_JOB_ID} ]; do sleep 1; done
    fi

    cd '"${REPO_DIR}"'

    python -m audio_tokenization.tokenize \
        dataset=audio_text/aishell_1_3_direct \
        mode=audio_text \
        audio_text_format=direct \
        num_gpus='"${TOTAL_GPUS}"' \
        resume='"${RESUME}"' \
        wandb.name="aishell_1_3_direct_tok_'"${SLURM_JOB_NUM_NODES}"'nodes" \
        wandb.tags="[tokenization,aishell,1_3,direct,'"${SLURM_JOB_NUM_NODES}"'nodes]"
'

echo "=========================================="
echo "AISHELL-1+3 direct tokenization finished"
echo "=========================================="
