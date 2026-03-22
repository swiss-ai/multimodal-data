#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=tok-legco-itlv
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/legco-speech/logs/tok_legco_interleaved_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/legco-speech/logs/tok_legco_interleaved_%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=2
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
if [ "${RESUME}" = "resume" ]; then
    RESUME=true
fi

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
LOG_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/legco-speech/logs
mkdir -p "${LOG_DIR}"

TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * 4))

echo "=========================================="
echo "LegCo Speech Tokenization (Interleaved)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Total GPUs: ${TOTAL_GPUS}"
echo "Resume: ${RESUME}"
echo "=========================================="

srun --kill-on-bad-exit=0 -u bash -c '
    set -e
    export LD_PRELOAD=/opt/venv/lib/python3.12/site-packages/ray/core/libjemalloc.so
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:'"${REPO_DIR}"':${PYTHONPATH:-}

    # Install torchaudio once per node; other local ranks wait.
    if [ "${SLURM_LOCALID}" = "0" ]; then
        uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation \
            git+https://github.com/pytorch/audio.git@release/2.9
        touch /tmp/.torchaudio_ready_${SLURM_JOB_ID}
    else
        while [ ! -f /tmp/.torchaudio_ready_${SLURM_JOB_ID} ]; do sleep 1; done
    fi

    cd '"${REPO_DIR}"'

    python -m audio_tokenization.tokenize \
        dataset=audio_text/legco_speech_train_interleaved \
        mode=audio_text \
        audio_text_format=interleaved \
        num_gpus='"${TOTAL_GPUS}"' \
        resume='"${RESUME}"' \
        wandb.name="legco_interleaved_tok_'"${SLURM_JOB_NUM_NODES}"'nodes" \
        wandb.tags="[tokenization,legco,interleaved,'"${SLURM_JOB_NUM_NODES}"'nodes]"
'

echo "=========================================="
echo "LegCo interleaved tokenization finished"
echo "=========================================="
