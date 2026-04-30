#!/bin/bash
# Catalan ParlamentParla v3 → Lhotse Shar (~1060h, all train splits)
# Audio: tar.gz archives with wav files
# Metadata: CSV with identifier,segment_path,text
#SBATCH --account=infra01
#SBATCH --job-name=catalan-shar
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/convert/catalan-shar-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/convert/catalan-shar-%x-%A.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

ulimit -n 65536

uv pip install --python /opt/venv/bin/python orjson

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer:${PYTHONPATH:-}
export FFMPEG_ROOT=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64
export PATH=${FFMPEG_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}

RAW_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___projecte-aina___parlament_parla_v3
SHAR_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/parlament_parla
TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

echo "=========================================="
echo "ParlamentParla v3 → Shar"
echo "=========================================="

# Step 1: Merge all train CSVs into one CSV
METADATA=/tmp/parlament_parla_metadata.csv
head -1 ${RAW_DIR}/corpus/files/clean_train_parlament_short.csv > ${METADATA}
for csv in \
    ${RAW_DIR}/corpus/files/clean_train_parlament_short.csv \
    ${RAW_DIR}/corpus/files/clean_train_parlament_long.csv \
    ${RAW_DIR}/corpus/files/other_train_parlament_short.csv \
    ${RAW_DIR}/corpus/files/other_train_parlament_long.csv; do
    tail -n +2 "$csv" >> ${METADATA}
done
echo "Metadata: $(wc -l < ${METADATA}) rows"

# Step 2: Collect all train tar.gz paths
TARS=$(find \
    ${RAW_DIR}/corpus/speech/clean_train_short/ \
    ${RAW_DIR}/corpus/speech/clean_train_long/ \
    ${RAW_DIR}/corpus/speech/other_train_short/ \
    ${RAW_DIR}/corpus/speech/other_train_long/ \
    -name "*.tar.gz" | sort | tr '\n' ' ')

echo "Found tars: $(echo $TARS | wc -w)"

# Step 3: Convert using prepare_wds_to_shar
python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
    --wds-shards ${TARS} \
    --external-metadata "${METADATA}" \
    --id-field identifier \
    --text-field text \
    --shar-dir "${SHAR_ROOT}" \
    --target-sr 24000 \
    --shard-size 20000 \
    --text-tokenizer "${TOKENIZER}" \
    --num-workers 32

echo ""
echo "=========================================="
echo "ParlamentParla done!"
echo "=========================================="
