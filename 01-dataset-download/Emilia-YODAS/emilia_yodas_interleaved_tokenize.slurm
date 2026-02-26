#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ey-intlv-tok
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/ey_interleaved_tok_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/ey_interleaved_tok_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -e

# Usage:
#   sbatch scripts/emilia_yodas_interleaved_tokenize.slurm          # fresh run
#   sbatch scripts/emilia_yodas_interleaved_tokenize.slurm resume   # resume from checkpoints
RESUME=${1:-false}

echo "=========================================="
echo "Emilia-YODAS Interleaved Audio-Text Tokenization"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "GPUs per node: 4"
echo "=========================================="

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs

TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * 4))

echo "Total ranks: ${TOTAL_GPUS}"
echo "Resume: ${RESUME}"
echo ""

# 4 srun tasks per node (1 per GPU), no torchrun, no NCCL.
#   --kill-on-bad-exit=0 : one rank dying does NOT kill others
#   LD_PRELOAD=jemalloc  : prevents glibc malloc fragmentation RSS growth
srun --kill-on-bad-exit=0 -u bash -c '
    set -e
    export LD_PRELOAD=/opt/venv/lib/python3.12/site-packages/ray/core/libjemalloc.so
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer:$PYTHONPATH

    # Install torchaudio once per node; other local ranks wait.
    if [ "${SLURM_LOCALID}" = "0" ]; then
        uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation \
            git+https://github.com/pytorch/audio.git@release/2.9
        touch /tmp/.torchaudio_ready_${SLURM_JOB_ID}
    else
        while [ ! -f /tmp/.torchaudio_ready_${SLURM_JOB_ID} ]; do sleep 1; done
    fi

    cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer

    python -m audio_tokenization.tokenize \
        dataset=audio_text/emilia_yodas_interleaved \
        mode=audio_text \
        audio_text_format=interleaved \
        num_gpus='"${TOTAL_GPUS}"' \
        resume='"${RESUME}"' \
        wandb.name="emilia_yodas_interleaved_tok_'"${SLURM_JOB_NUM_NODES}"'nodes" \
        wandb.tags="[emilia_yodas,interleaved,audio_text,'"${SLURM_JOB_NUM_NODES}"'nodes]"
'

echo ""
echo "=========================================="
echo "Tokenization completed!"
echo "=========================================="
