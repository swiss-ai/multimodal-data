#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=tok-coral
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/coral/logs/tok_coral_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/coral/logs/tok_coral_%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

# Usage:
#   sbatch tokenize.slurm                         # read_aloud (direct)
#   sbatch tokenize.slurm conversation            # conversation (interleaved)
#   sbatch tokenize.slurm both                    # run read_aloud then conversation
#   sbatch tokenize.slurm read_aloud resume       # resume checkpoints
VARIANT=${VARIANT:-${1:-read_aloud}}
RESUME=${RESUME:-${2:-false}}
if [ "${RESUME}" = "resume" ]; then
    RESUME=true
fi

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
LOG_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/coral/logs
mkdir -p "${LOG_DIR}"

TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * 4))

echo "=========================================="
echo "CoRal v3 Tokenization"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Variant: ${VARIANT}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Total GPUs: ${TOTAL_GPUS}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Resume: ${RESUME}"
echo "=========================================="

run_variant() {
    local name="$1"
    local dataset_cfg="$2"
    local audio_text_format="$3"
    local srun_nodes="$4"
    local srun_ntasks="$5"

    echo ""
    echo "[${name}] dataset=${dataset_cfg} format=${audio_text_format} nodes=${srun_nodes} ntasks=${srun_ntasks}"

    srun --nodes="${srun_nodes}" --ntasks="${srun_ntasks}" --ntasks-per-node=4 --kill-on-bad-exit=0 -u bash -c '
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
            dataset='"${dataset_cfg}"' \
            mode=audio_text \
            audio_text_format='"${audio_text_format}"' \
            num_gpus='"${TOTAL_GPUS}"' \
            resume='"${RESUME}"' \
            wandb.name="coral_v3_'"${name}"'_tok_'"${SLURM_JOB_NUM_NODES}"'nodes" \
            wandb.tags="[tokenization,coral_v3,'"${name}"','"${audio_text_format}"','"${SLURM_JOB_NUM_NODES}"'nodes]"
    '
}

case "${VARIANT}" in
    read_aloud)
        # read_aloud has 4 SHAR shards; run with 4 ranks to avoid zero-cut ranks.
        run_variant "read_aloud" "audio_text/coral_v3_read_aloud_train_direct" "direct" "1" "4"
        ;;
    conversation)
        run_variant "conversation" "audio_text/coral_v3_conversation_train_interleaved" "interleaved" "${SLURM_JOB_NUM_NODES}" "${TOTAL_GPUS}"
        ;;
    both)
        run_variant "read_aloud" "audio_text/coral_v3_read_aloud_train_direct" "direct" "1" "4"
        run_variant "conversation" "audio_text/coral_v3_conversation_train_interleaved" "interleaved" "${SLURM_JOB_NUM_NODES}" "${TOTAL_GPUS}"
        ;;
    *)
        echo "ERROR: unknown VARIANT='${VARIANT}' (expected: read_aloud | conversation | both)" >&2
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "CoRal v3 tokenization finished"
echo "=========================================="
