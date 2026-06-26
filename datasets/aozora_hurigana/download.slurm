#!/bin/bash
# Download NDL Aozora Hurigana Speech Corpus v2 (~71GB, 4892h Japanese speech, CC0 / Public Domain)
# From National Diet Library (NDL), built from Aozora Bunko PD texts + Sapie audio daisy.
# https://github.com/ndl-lab/hurigana-speech-corpus-aozora
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-aozora-hurigana
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/aozora-hurigana-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/aozora-hurigana-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=SD-69241-apertus-1-5-5

set -euo pipefail

DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/ndl___aozora_hurigana_speech_v2"
CONNECTIONS=16

mkdir -p "${DEST_DIR}"

echo "========================================"
echo "NDL Aozora Hurigana Speech Corpus v2 Download"
echo "Dest: ${DEST_DIR}"
echo "========================================"

URL_FILE=$(mktemp)
cat > "${URL_FILE}" <<'EOF'
https://lab.ndl.go.jp/dataset/hurigana-speech-corpus-aozora/ver2/aozora_work_part1.zip
https://lab.ndl.go.jp/dataset/hurigana-speech-corpus-aozora/ver2/aozora_work_part2.zip
EOF

aria2c -x "${CONNECTIONS}" -s "${CONNECTIONS}" \
    --continue=true \
    --max-tries=10 \
    --retry-wait=10 \
    --summary-interval=30 \
    -d "${DEST_DIR}" \
    -i "${URL_FILE}"

rm -f "${URL_FILE}"

echo "[$(date '+%F %T')] Download finished"
echo "Final size:"
du -sh "${DEST_DIR}"
echo "Done."
