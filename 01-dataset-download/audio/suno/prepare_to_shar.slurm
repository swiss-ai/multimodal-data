#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=suno-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/suno/logs/suno_shar_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/suno/logs/suno_shar_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer:${PYTHONPATH:-}

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/suno/logs

echo "Suno shar conversion | Job: ${SLURM_JOB_ID} | Node: $(hostname)"

# S1: audio-only (no gpt description prompt)
echo "=== S1 (audio-only) ==="
python -m audio_tokenization.prepare.prepare_wds_to_shar \
    --wds-shards '/capstor/store/cscs/swissai/infra01/audio-datasets/raw/suno/shards_s1/*.tar' \
    --shar-dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1_rebuild/suno_s1_shar_rmsdb_v1 \
    --target-sr 24000 \
    --min-sr 16000 \
    --shard-size 5000 \
    --shar-format flac \
    --resampling-backend torchaudio \
    --num-workers 128

# S2: music caption (audio + gpt description prompt as .txt). Disabled by
# default because this launcher is now used for the stage-1 rebuild path.
RUN_S2=${RUN_S2:-0}
if [ "${RUN_S2}" = "1" ]; then
    echo "=== S2 (music caption) ==="
    python -m audio_tokenization.prepare.prepare_wds_to_shar \
        --wds-shards '/capstor/store/cscs/swissai/infra01/audio-datasets/raw/suno/shards_s2/*.tar' \
        --shar-dir /capstor/store/cscs/swissai/infra01/audio-datasets/music_caption_shar/suno_s2_shar \
        --target-sr 24000 \
        --min-sr 16000 \
        --shard-size 5000 \
        --shar-format flac \
        --resampling-backend torchaudio \
        --num-workers 128
fi

echo "Suno shar conversion finished"
