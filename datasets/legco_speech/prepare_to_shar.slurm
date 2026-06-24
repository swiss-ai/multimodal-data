#!/bin/bash
# Convert LegCo-Speech (Cantonese) parquet to Lhotse Shar format
# Job array: each array task runs on a separate node processing a chunk of parquet files
#SBATCH --account=infra01
#SBATCH --job-name=legco-shar
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/legco-speech/logs/%x-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/legco-speech/logs/%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL
#SBATCH --exclusive

set -euo pipefail

ulimit -n 65536

uv pip install --python /opt/venv/bin/python polars

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer:${PYTHONPATH:-}
export FFMPEG_ROOT=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64
export PATH=${FFMPEG_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}
MP_START_METHOD=${MP_START_METHOD:-fork}

PARQUET_SRC=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/legco-speech/segmented
SHAR_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/legco_speech_train

RANK="${SLURM_ARRAY_TASK_ID}"
WORLD=4

echo "LegCo-Speech Shar conversion | Job: ${SLURM_JOB_ID} | Array: ${RANK}/${WORLD} | Node: $(hostname)"
echo "mp_start_method: ${MP_START_METHOD}"
/opt/venv/bin/python -c "import shutil; print('ffmpeg:', shutil.which('ffmpeg'))"

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

echo "Array task ${RANK}: processing parquets [${START}..$(( START + COUNT ))) out of ${TOTAL}"

# Create temp dir with symlinks to this task's parquet files
TMPDIR_PQ=$(mktemp -d)
for (( i=START; i<START+COUNT; i++ )); do
    ln -s "${ALL_PQ[$i]}" "${TMPDIR_PQ}/"
done

# Each array task writes to its own subdirectory
NODE_TAG="$(printf "%02d" "${RANK}")"
NODE_SHAR_DIR="${SHAR_ROOT}/node_${NODE_TAG}"
mkdir -p "${NODE_SHAR_DIR}"

/opt/venv/bin/python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${TMPDIR_PQ}" \
    --shar-dir "${NODE_SHAR_DIR}" \
    --id-column video_id segment_id \
    --target-sr 24000 \
    --text-tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json \
    --shard-size 5000 \
    --num-workers 128 \
    --mp-start-method "${MP_START_METHOD}" \
    --ffmpeg-bin ffmpeg

rm -rf "${TMPDIR_PQ}"

echo "LegCo-Speech Shar conversion finished on array task ${RANK}"
