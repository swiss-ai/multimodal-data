#!/bin/bash
# Convert uhhlt/Tuda-De arrow → SHAR (cooldown). Train split only, 55,500 samples.
#SBATCH --account=infra01
#SBATCH --job-name=tuda-de-to-shar
#SBATCH --reservation=SD-69241-apertus-1-5-6
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/tuda-de-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/tuda-de-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}
LHOTSE_REPO=${LHOTSE_REPO:-/iopsstor/scratch/cscs/xyixuan/dev/lhotse}
BENCHMARK_REPO=${BENCHMARK_REPO:-/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer}

ARROW_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___uhhlt___Tuda-De/splits/train"
SHAR_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/cooldown/tuda_de_train"

FFMPEG_ROOT=${FFMPEG_ROOT:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64}
TEXT_TOKENIZER=${TEXT_TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json}

NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-288}}

export PATH="${FFMPEG_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${LHOTSE_REPO}:${BENCHMARK_REPO}:${PYTHONPATH:-}"
uv pip install --python /opt/venv/bin/python polars

echo "=========================================="
echo "Tuda-De -> SHAR (cooldown)"
echo "Arrow:   ${ARROW_DIR}"
echo "SHAR:    ${SHAR_DIR}"
echo "Workers: ${NUM_WORKERS}"
echo "=========================================="

"${PYTHON_BIN}" -m audio_tokenization.utils.prepare_data.prepare_hf_to_shar \
    --arrow-dir "${ARROW_DIR}" \
    --arrow-glob "data-*.arrow" \
    --shar-dir "${SHAR_DIR}" \
    --id-column sentence_id \
    --text-column sentence \
    --custom-columns gender microphone speaker_id angle ageclass cleaned_sentence corpus muttersprachler bundesland source \
    --target-sr 24000 \
    --text-tokenizer "${TEXT_TOKENIZER}" \
    --shard-size 5000 \
    --num-workers "${NUM_WORKERS}"

echo "[$(date '+%F %T')] Done"
