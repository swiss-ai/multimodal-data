#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=tok-gigaspeech2
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/tok-gigaspeech2_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/tok-gigaspeech2_%j.err
#SBATCH --time=12:00:00
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
mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs

TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * 4))

echo "=========================================="
echo "GigaSpeech2 Direct Tokenization"
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
        dataset=audio_text/gigaspeech2_th_id_vi_direct \
        mode=audio_text \
        audio_text_format=direct \
        audio_text_task=transcribe \
        num_gpus='"${TOTAL_GPUS}"' \
        resume='"${RESUME}"' \
        wandb.enabled=true \
        wandb.name="gigaspeech2_th_id_vi_direct_1node" \
        wandb.tags="[tokenization,gigaspeech2,direct,1node]"
'

echo "=========================================="
echo "GigaSpeech2 direct tokenization finished"
echo "=========================================="
