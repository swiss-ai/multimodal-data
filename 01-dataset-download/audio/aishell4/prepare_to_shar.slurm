#!/bin/bash
# AISHELL-4 → Lhotse Shar (120h Mandarin meeting speech, 3 splits)
#SBATCH --account=infra01
#SBATCH --job-name=aishell4-shar
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/convert/aishell4-shar-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/convert/aishell4-shar-%x-%A.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

ulimit -n 65536

uv pip install --python /opt/venv/bin/python textgrid

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer:${PYTHONPATH:-}

CORPUS_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/aishell/aishell4
SHAR_BASE=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/aishell
TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

echo "=========================================="
echo "AISHELL-4 → Shar (all train splits)"
echo "=========================================="

for split in train_L train_M train_S; do
    echo "--- Converting ${split} ---"

    python -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
        --recipe aishell4 \
        --corpus_dir "${CORPUS_DIR}" \
        --split "${split}" \
        --recipe_kwargs '{"normalize_text": true}' \
        --trim_to_supervisions \
        --shar_output_dir "${SHAR_BASE}/aishell4_${split}" \
        --target_sample_rate 24000 \
        --text_tokenizer "${TOKENIZER}" \
        --shar_shard_size 5000 \
        --num_workers 64
done

echo ""
echo "=========================================="
echo "AISHELL-4 Shar conversion completed!"
echo "=========================================="
