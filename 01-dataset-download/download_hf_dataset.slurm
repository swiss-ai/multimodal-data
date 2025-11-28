#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --environment=emu3
#SBATCH --job-name=dwnldHFdset
#SBATCH --output=/iopsstor/scratch/cscs/%u/multimodal-data/01-dataset-download/logs/hf-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/multimodal-data/01-dataset-download/logs/hf-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1

################################################################################
# HuggingFace Dataset Downloader
################################################################################
# Downloads HF datasets to cache for tokenization pipeline.
# Use --cache-dir during tokenization to load from cache.
#
# USAGE:
#   sbatch download_dataset_hf.slurm <subset>
#   DATASET_NAME="google/docci" sbatch download_dataset_hf.slurm ""
#
# PARAMETERS (environment variables):
#   DATASET_NAME     - HF dataset repo (default: HuggingFaceM4/FineVision)
#   SUBSET_NAME      - Dataset subset/config, single or comma-separated list
#                      (default: $1, use "" for auto-detect all configs)
#   SPLIT            - Dataset split (default: train) - NOTE: Currently ignored,
#                      all splits are downloaded by download_and_prepare()
#   CACHE_DIR        - Datasets cache path for processed datasets
#                      (default: /capstor/.../hf_cache)
#   HF_HUB_CACHE     - HuggingFace Hub cache path for raw downloads
#                      (default: ~/.cache/huggingface/hub)
#                      IMPORTANT: Set this for large datasets or when using
#                      central cluster cache to avoid filling home directory
#   NUM_PROC         - Download processes (default: auto = CPUs available)
#   FORCE_REDOWNLOAD - Re-download if cached (default: false)
#   MAX_RETRIES      - Maximum retry attempts on failure (default: 10)
#   BACKOFF_FACTOR   - Exponential backoff multiplier (default: 1.2)
#   USE_HF_TRANSFER  - Use hf_transfer for faster downloads (default: false)
#                      WARNING: Bypasses HTTP retry logic on download/resolve
#   CLUSTER_REPO_HOME - defaults to $SLURM_SUBMIT_DIR, must point to location of repository in the cluster env
#
# EXAMPLES:
#   sbatch download_dataset_hf.slurm laion_gpt4v
#   DATASET_NAME="google/docci" sbatch download_dataset_hf.slurm ""
#   SUBSET_NAME="ParaphraseRC,SelfRC" sbatch download_dataset_hf.slurm
#   CACHE_DIR="/custom/path" sbatch download_dataset_hf.slurm ocrvqa
#   HF_HUB_CACHE="/capstor/cache/hf_hub" sbatch download_dataset_hf.slurm ocrvqa
#   NUM_PROC=64 sbatch download_dataset_hf.slurm lvis_instruct4v
#   MAX_RETRIES=20 BACKOFF_FACTOR=1.5 sbatch download_dataset_hf.slurm ocrvqa
#   USE_HF_TRANSFER=true sbatch download_dataset_hf.slurm ocrvqa
################################################################################

# ========================================
# Configuration Parameters
# ========================================

# Dataset configuration (overridable via environment variables)
DATASET_NAME="${DATASET_NAME:-mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M}"
SUBSET_NAME="${SUBSET_NAME:-$1}"  # Use first argument if SUBSET_NAME not set
SPLIT="${SPLIT:-train}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/huggingface/datasets}"
FORCE_REDOWNLOAD="${FORCE_REDOWNLOAD:-false}"
NUM_PROC="${NUM_PROC:-auto}"
MAX_RETRIES="${MAX_RETRIES:-10}"
BACKOFF_FACTOR="${BACKOFF_FACTOR:-1.2}"
USE_HF_TRANSFER="${USE_HF_TRANSFER:-false}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HOME}/.cache/huggingface/hub}"
CLUSTER_REPO_HOME="${CLUSTER_MOME:-$SLURM_SUBMIT_DIR}" # directory where to find this repository

# Auto-detect number of processes if set to "auto"
if [ "$NUM_PROC" = "auto" ]; then
    # Use SLURM CPUs if available, otherwise detect from system
    if [ -n "$SLURM_CPUS_PER_TASK" ]; then
        TOTAL_CPUS=$SLURM_CPUS_PER_TASK
    else
        TOTAL_CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 32)
    fi
    # Set to half the number of CPUs, minimum 1
    NUM_PROC=$((TOTAL_CPUS / 2))
    if [ "$NUM_PROC" -lt 1 ]; then
        NUM_PROC=1
    fi
fi

# ========================================
# Display Configuration
# ========================================

echo "=========================================="
echo "HuggingFace Dataset Download Job"
echo "=========================================="
echo "Repo-Root: $CLUSTER_REPO_HOME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $NUM_PROC"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Dataset:          $DATASET_NAME"
echo "  Subset:           ${SUBSET_NAME:-<none>}"
echo "  Split:            $SPLIT"
echo ""
echo "Cache Locations:"
echo "  Datasets Cache:   $CACHE_DIR"
echo "                    (Processed datasets, ready for use)"
echo "  HF Hub Cache:     $HF_HUB_CACHE"
echo "                    (Raw downloads from HuggingFace Hub)"
echo ""
echo "Download Settings:"
echo "  Num Processes:    $NUM_PROC"
echo "  Force Redownload: $FORCE_REDOWNLOAD"
echo "  Max Retries:      $MAX_RETRIES"
echo "  Backoff Factor:   $BACKOFF_FACTOR"
echo "  Use HF Transfer:  $USE_HF_TRANSFER"
echo "=========================================="

# Create cache directory
mkdir -p "$CACHE_DIR"
# Create log directory
mkdir -p $CLUSTER_REPO_HOME/01-dataset-download/logs
# Set threading per process for openMP enabled libs
export OMP_NUM_THREADS=1

# ========================================
# Python Environment Setup
# ========================================

echo ""
echo "Setting up Python environment..."

# Install all required packages from requirements.txt
echo "Installing required packages from requirements.txt..."
pip install -q -r "${CLUSTER_REPO_HOME}/01-dataset-download/requirements.txt"

if [ $? -eq 0 ]; then
    echo "  ✓ All packages installed successfully"
else
    echo "  ✗ Package installation failed!"
    exit 1
fi

echo "Environment setup complete"

# Convert hf datasets identifier to cache name
hf_dataset_to_cache_name() {
    python3 -c "from datasets.naming import camelcase_to_snakecase as c; n='$1'; print((n.split('/')[0]+'___'+c(n.split('/')[1])) if '/' in n else c(n))"
}

get_hf_dataset_cache_path() {
    local dataset_name="$1"
    local subset_name="$2"
    local cache_dir="$3"

    local sanitized_name=$(hf_dataset_to_cache_name "$dataset_name")

    if [ -n "$subset_name" ]; then
        echo "${cache_dir}/${sanitized_name}/${subset_name}"
    else
        echo "${cache_dir}/${sanitized_name}"
    fi
}

echo ""
echo "Cleaning up stale lock files for this dataset..."

DATASET_CACHE_PATH=$(get_hf_dataset_cache_path "$DATASET_NAME" "$SUBSET_NAME" "$CACHE_DIR")
echo "Dataset cache path: $DATASET_CACHE_PATH"

# Check if the dataset cache directory exists
if [ -d "$DATASET_CACHE_PATH" ]; then
    LOCK_COUNT=$(find "$DATASET_CACHE_PATH" -name "*.lock" 2>/dev/null | wc -l)
    if [ "$LOCK_COUNT" -gt 0 ]; then
        echo "Found $LOCK_COUNT lock file(s) in dataset directory - removing..."
        find "$DATASET_CACHE_PATH" -name "*.lock" -delete 2>/dev/null
        echo "Lock files cleaned up"
    else
        echo "No stale lock files found in dataset directory"
    fi
else
    echo "⚠ Dataset cache directory does not exist yet (first download)"
    echo "  Skipping lock file cleanup"
fi

# Configure HF transfer and caching
if [ "$USE_HF_TRANSFER" = "true" ]; then
    echo ""
    echo "⚠️  WARNING: HF_TRANSFER is ENABLED"
    echo "    This may speed up large file downloads but BYPASSES:"
    echo "    - HTTP retry logic on download/resolve operations"
    echo "    - Custom backoff and rate limit handling"
    echo "    Downloads may fail faster on network issues."
    echo ""
    export HF_HUB_ENABLE_HF_TRANSFER=1
else
    export HF_HUB_ENABLE_HF_TRANSFER=0
fi

# Xet Cache does not have benefit on first time download
unset HF_HUB_ENABLE_XET_CACHE

# Export HF_HUB_CACHE so Python scripts inherit it
export HF_HUB_CACHE


# ========================================
# Download Dataset using Builder
# ========================================
echo ""
echo "Starting dataset download..."
echo ""

PYTHON_CMD="python3 $CLUSTER_REPO_HOME/01-dataset-download/download_hf_dataset.py \
    --dataset-name \"${DATASET_NAME}\" \
    --cache-dir \"${CACHE_DIR}\" \
    --num-proc ${NUM_PROC} \
    --max-retries ${MAX_RETRIES} \
    --backoff-factor ${BACKOFF_FACTOR}"

# Add optional subset name if provided
if [ -n "$SUBSET_NAME" ]; then
    PYTHON_CMD="${PYTHON_CMD} --subset-name \"${SUBSET_NAME}\""
fi

# Add force redownload flag if enabled
if [ "$FORCE_REDOWNLOAD" = "true" ]; then
    PYTHON_CMD="${PYTHON_CMD} --force-redownload"
fi

# Execute the Python script
eval "$PYTHON_CMD"

EXIT_CODE=$?

# ========================================
# Job Summary
# ========================================

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "SUCCESS! Dataset cached and ready for tokenization."
    echo ""
    echo "Cache location: $CACHE_DIR"
    echo "Cache size: $(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1 || echo 'unknown')"
    echo ""
    echo "Use with tokenization via:"
    echo "  --cache-dir $CACHE_DIR"
    echo ""
else
    echo ""
    echo "ERROR: Download failed with exit code: $EXIT_CODE"
    echo "Check error logs for details."
    echo ""
fi

echo "=========================================="