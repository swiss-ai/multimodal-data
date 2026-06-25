#!/bin/bash
# Convert LegCo-Speech (Cantonese) parquet to Lhotse Shar format with universal IDs
# Job array: each array task processes a contiguous chunk of parquet files
#SBATCH --account=infra01
#SBATCH --job-name=legco-shar
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/legco-shar-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/legco-shar-%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL
#SBATCH --exclusive

set -euo pipefail

ulimit -n 65536

REPO=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
PARQUET_SRC=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/legco-speech/segmented
SHAR_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/legco_speech_train_v2

RANK="${SLURM_ARRAY_TASK_ID}"
WORLD="${SLURM_ARRAY_TASK_COUNT}"

source "${REPO}/scripts/source_lhotse_runtime.sh"
uv pip install --python /opt/venv/bin/python polars

# Distribute parquet files across array tasks (contiguous chunks)
mapfile -t ALL_PQ < <(ls "${PARQUET_SRC}"/*.parquet | sort)
TOTAL=${#ALL_PQ[@]}
BASE=$(( TOTAL / WORLD ))
REM=$(( TOTAL % WORLD ))
if [ "${RANK}" -lt "${REM}" ]; then
    COUNT=$(( BASE + 1 ))
    START=$(( RANK * COUNT ))
else
    COUNT=${BASE}
    START=$(( REM * (BASE + 1) + (RANK - REM) * BASE ))
fi

echo "=========================================="
echo "LegCo Speech → Shar (rank ${RANK}/${WORLD})"
echo "  Node: $(hostname -s)"
echo "  Parquets: [${START}..$(( START + COUNT ))) of ${TOTAL}"
echo "=========================================="

# Symlink this task's parquet slice into a temp dir
TMPDIR_PQ=$(mktemp -d)
trap "rm -rf ${TMPDIR_PQ}" EXIT
for (( i=START; i<START+COUNT; i++ )); do
    ln -s "${ALL_PQ[$i]}" "${TMPDIR_PQ}/"
done

NODE_TAG="$(printf "%02d" "${RANK}")"
NODE_SHAR_DIR="${SHAR_ROOT}/node_${NODE_TAG}"
mkdir -p "${NODE_SHAR_DIR}"

python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${TMPDIR_PQ}" \
    --shar-dir "${NODE_SHAR_DIR}" \
    --id-column video_id segment_id \
    --audio-column audio \
    --text-column text \
    --duration-column duration \
    --input-clip-id-parser trailing_number \
    --text-tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json \
    --target-sr 24000 \
    --shard-size 20000 \
    --num-workers 32

echo "Rank ${RANK} completed!"

# Last task submits aggregate job
if [ "${RANK}" -eq "$((WORLD - 1))" ]; then
    echo "Submitting aggregate job..."
    sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} \
        --account=infra01 --partition=normal \
        --environment=nemo_25_11 \
        --time=00:10:00 --nodes=1 --cpus-per-task=1 \
        --job-name=legco-agg \
        --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/legco-agg-%j.out \
        --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/legco-agg-%j.err \
        --wrap="source ${REPO}/scripts/source_lhotse_runtime.sh; \
                python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
                    --aggregate ${SHAR_ROOT} \
                    --text-tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json"
    echo "Aggregate job submitted"
fi
