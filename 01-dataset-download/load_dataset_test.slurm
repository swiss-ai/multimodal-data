#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --environment=emu3
#SBATCH --job-name=testHFdset
#SBATCH --output=/iopsstor/scratch/cscs/%u/multimodal-data/01-dataset-download/logs/test-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/multimodal-data/01-dataset-download/logs/test-%x-%A.err
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1

################################################################################
# HuggingFace Dataset Loader Test
################################################################################
# Tests loading of downloaded HuggingFace datasets with comprehensive statistics
#
# USAGE:
#   sbatch load_dataset_test.slurm
#   DATASET_NAME="mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M" SPLIT="train[:10%]" NUM_PROC="8" METHOD="default" STREAMING=false sbatch 01-dataset-download/load_dataset_test.slurm
#   DATASET_NAME="mvp-lab/LLaVA-OneVision-1.5-Instruct-Data" SPLIT="train" NUM_PROC="8" METHOD="default" STREAMING=false sbatch 01-dataset-download/load_dataset_test.slurm
#
# PARAMETERS (environment variables):
#   DATASET_NAME     - HF dataset repo (default: mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M)
#   SUBSET_NAME      - Dataset subset/config (optional)
#   SPLIT            - Dataset split (default: train)
#   CACHE_DIR        - Datasets cache path for processed datasets
#                      (default: /capstor/.../hf_datasets_cache)
#   HF_HUB_CACHE     - HuggingFace Hub cache path for raw downloads
#                      (default: /capstor/.../hf_hub_cache)
#   METHOD           - Loading method: "default" or "builder_load" (default: builder_load)
#   STREAMING        - Enable streaming (default: false)
#   NUM_PROC         - Number of processes (default: auto)
#   CLUSTER_REPO_HOME - Repository location (default: $SLURM_SUBMIT_DIR)
################################################################################

# ========================================
# Configuration Parameters
# ========================================

# Dataset configuration (overridable via environment variables)
DATASET_NAME="${DATASET_NAME:-mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M}"
SUBSET_NAME="${SUBSET_NAME:-}"
SPLIT="${SPLIT:-train}"
CACHE_DIR="${CACHE_DIR:-/capstor/store/cscs/swissai/infra01/vision-datasets/hf_datasets_cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-/capstor/store/cscs/swissai/infra01/vision-datasets/hf_hub_cache}"
METHOD="${METHOD:-builder_load}"
STREAMING="${STREAMING:-false}"
NUM_PROC="${NUM_PROC:-auto}"
CLUSTER_REPO_HOME="${CLUSTER_REPO_HOME:-$SLURM_SUBMIT_DIR}"

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
echo "HuggingFace Dataset Loader Test Job"
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
echo "  Method:           $METHOD"
echo "  Streaming:        $STREAMING"
echo ""
echo "Cache Locations:"
echo "  Datasets Cache:   $CACHE_DIR"
echo "                    (Processed datasets, ready for use)"
echo "  HF Hub Cache:     $HF_HUB_CACHE"
echo "                    (Raw downloads from HuggingFace Hub)"
echo ""
echo "Test Settings:"
echo "  Num Processes:    $NUM_PROC"
echo "=========================================="

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

# Export HF_HUB_CACHE so Python scripts inherit it
export HF_HUB_CACHE

# ========================================
# Run Dataset Loader Test
# ========================================
echo ""
echo "Starting dataset loader test..."
echo ""

PYTHON_CMD="python3 $CLUSTER_REPO_HOME/01-dataset-download/load_dataset_test.py \
    --dataset-name \"${DATASET_NAME}\" \
    --cache-dir \"${CACHE_DIR}\" \
    --method ${METHOD} \
    --split \"${SPLIT}\""

# Add optional subset name if provided
if [ -n "$SUBSET_NAME" ]; then
    PYTHON_CMD="${PYTHON_CMD} --subset-name \"${SUBSET_NAME}\""
fi

# Add streaming flag if enabled
if [ "$STREAMING" = "true" ]; then
    PYTHON_CMD="${PYTHON_CMD} --streaming"
fi

# Add num-proc if not using builder_load method
if [ "$METHOD" != "builder_load" ] && [ "$NUM_PROC" != "auto" ]; then
    PYTHON_CMD="${PYTHON_CMD} --num-proc ${NUM_PROC}"
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
    echo "SUCCESS! Dataset test completed."
    echo ""
    echo "Cache location: $CACHE_DIR"
    echo ""
else
    echo ""
    echo "ERROR: Test failed with exit code: $EXIT_CODE"
    echo "Check error logs for details."
    echo ""
fi

echo "=========================================="