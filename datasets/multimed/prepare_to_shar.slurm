#!/bin/bash
# MultiMed → Lhotse Shar (137h, 5 languages, medical speech)
#SBATCH --account=infra01
#SBATCH --job-name=multimed-shar
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/convert/multimed-shar-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/convert/multimed-shar-%x-%A.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

ulimit -n 65536

uv pip install --python /opt/venv/bin/python polars

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer:${PYTHONPATH:-}
export FFMPEG_ROOT=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64
export PATH=${FFMPEG_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}

RAW_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/multimed
SHAR_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/multimed
TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

echo "=========================================="
echo "MultiMed → Shar (per language)"
echo "=========================================="

# Chinese  → zh
# English  → en
# French   → fr
# German   → de
# Vietnamese → vi

for pair in Chinese:zh English:en French:fr German:de Vietnamese:vi; do
    lang_name="${pair%%:*}"
    lang_code="${pair##*:}"

    echo "--- ${lang_name} (${lang_code}) ---"

    python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
        --parquet-dir "${RAW_DIR}/${lang_name}" \
        --parquet-glob '*.parquet' \
        --shar-dir "${SHAR_ROOT}/${lang_code}_train" \
        --audio-column audio \
        --text-column text \
        --duration-column duration \
        --target-sr 24000 \
        --text-tokenizer "${TOKENIZER}" \
        --shard-size 5000 \
        --num-workers 128
done

echo ""
echo "=========================================="
echo "MultiMed done!"
echo "=========================================="
