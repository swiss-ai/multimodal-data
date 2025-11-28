# Dataset Download Scripts

Tools for downloading and verifying HuggingFace datasets with robust error handling and caching support.

## Dependencies

All required Python packages are specified in `requirements.txt` with minimum version requirements:

```bash
# Install dependencies locally
pip install -r requirements.txt
```

**Key dependencies:**
- `huggingface_hub>=0.16.0,<1.0.0` - **CRITICAL**: Must be <1.0.0 (configure_http_backend removed in 1.0.0)
- `datasets>=2.14.0` - Dataset loading and preparation
- `hf_transfer>=0.1.0` - Optional: faster downloads
- `requests>=2.28.0` - HTTP session with security fixes
- `urllib3>=1.26.0` - Retry mechanism
- `tqdm>=4.64.0` - Progress bars for cache verification

**Note**: The SLURM script automatically installs these dependencies at runtime using `requirements.txt`.

## Scripts Overview

### 1. `download_hf_dataset.py`

Downloads and processes HuggingFace datasets to a local cache with HTTP retry logic and automatic configuration detection.
It can deal with hf hub api rate limits, with its retry logic. Upon failure you can just restart the script and it will resume,
skipping already-downloaded configs.

**What it does:**
- Downloads datasets from HuggingFace Hub with configurable retry logic
- If subset not specified - Auto-detects and downloads all dataset configurations
- Skips already-cached configurations automatically
- Supports parallel downloads with configurable workers
- Provides detailed error reports for failed downloads

**Key options:**
- `--dataset-name`: HuggingFace dataset repository (required)
- `--subset-name`: Specific config(s) to download (comma-separated) or auto-detect all if omitted
- `--cache-dir`: Local cache directory path (required) - this directory is the hf datasets cache location where the processed dataset files which are rdy will be stored. Don't mix this up with the hf hub cache.
- `--num-proc`: Number of parallel download processes (default: auto-detect)
- `--max-retries`: Maximum retry attempts on failure (default: 5)
- `--backoff-factor`: Exponential backoff multiplier (default: 1.0)
- `--force-redownload`: Force re-download even if cached

**Cache Configuration:**

The download process uses **two separate cache locations** both can be configured:

1. **HF Hub Cache (`HF_HUB_CACHE`)**:
   - Stores raw files downloaded from HuggingFace Hub
   - Default: `~/.cache/huggingface/hub`
   - These are the original files before any processing

2. **Datasets Cache (`CACHE_DIR`)**:
   - Stores processed datasets ready for use
   - Must be specified via `--cache-dir` parameter
   - These are the files after `download_and_prepare()` processing

**Consideration son clusters like clariden**
- Your team might have a central cache location for datasets and the hf hub files. Set the cache paths accordingly.
- Especially for large files and datasets its recommended to use a cache location on the cluster filesystems (ex. capstor on alps)
- Ex. for vision datasets we use:
  - `CACHE_DIR=/capstor/store/cscs/swissai/infra01/vision-datasets/hf_datasets_cache` 
  - `HF_HUB_CACHE=/capstor/store/cscs/swissai/infra01/vision-datasets/hf_hub_cache`

**Setting the cache location example:**
```bash
# Set custom hub cache location for large downloads
export HF_HUB_CACHE="/capstor/cache/hf_hub"
python download_hf_dataset.py --dataset-name "..." --cache-dir "/path/to/cache"

# Or inline:
HF_HUB_CACHE="/capstor/cache/hf_hub" python download_hf_dataset.py ...
```

If `HF_HUB_CACHE` is not set, HuggingFace libraries default to `~/.cache/huggingface/hub`.

**Tips/Useful Knowledge:**
- When network is unstable (use higher `--max-retries` and `--backoff-factor`)
- Hf datasets are first downloaded by the huggingface hub and then processed by hf datasets library.
- If an error occurs during processing after download, it might be that the cache is corrupted (can happen especially with large datasets on the distributed filesystem)
- In such case, run the hf_hub_cache_check provided to make sure the cache is valid
- The retry logic doesn't overwhelm the hub api, it respects retry-after headers and waits for the specified time before retrying - you will not be blocked ;)

**Example:**
```bash
# Download all configs
python download_hf_dataset.py \
    --dataset-name "HuggingFaceM4/FineVision" \
    --cache-dir "/path/to/cache"

# Download specific configs with retry tuning
python download_hf_dataset.py \
    --dataset-name "ibm-research/duorc" \
    --subset-name "ParaphraseRC,SelfRC" \
    --cache-dir "./cache" \
    --max-retries 10 \
    --backoff-factor 1.5
```

---

### 2. `download_hf_dataset.slurm`

SLURM wrapper for `download_hf_dataset.py` with environment setup and job management.
This script is specific to Alps cluster so best check the paths and configuration before you run.

**What it does:**
- Sets up Python environment with required packages (datasets, hf_transfer, huggingface_hub)
- Configures SLURM job parameters and resource allocation
- Cleans up stale lock files before download
- Provides detailed job logging and progress tracking

**Configuration via environment variables:**
- `DATASET_NAME`: HF dataset repo (default: mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M)
- `SUBSET_NAME`: Dataset config(s) or use `""` to auto-detect all
- `CACHE_DIR`: Datasets cache path for processed datasets (default: /capstor/store/cscs/swissai/infra01/vision-datasets/hf_cache)
- `HF_HUB_CACHE`: HuggingFace Hub cache path for raw downloads (default: ~/.cache/huggingface/hub) - IMPORTANT for large datasets
- `NUM_PROC`: Download workers (default: auto = half of CPUs)
- `MAX_RETRIES`: Max retry attempts for each specific http download request (default: 10)
- `BACKOFF_FACTOR`: Backoff multiplier (default: 1.2)
- `FORCE_REDOWNLOAD`: Re-download if cached (default: false)
- `USE_HF_TRANSFER`: Enable fast transfer of actual downloads (default: false, bypasses retry logic)
- `CLUSTER_REPO_HOME`: Defaults to `SLURM_SUBMIT_DIR`, must point to location of this repository in the cluster environment

**Example:**
```bash
# Use defaults
sbatch download_hf_dataset.slurm

# Override dataset
DATASET_NAME="google/docci" sbatch download_hf_dataset.slurm ""

# Set custom hub cache for large datasets
HF_HUB_CACHE="/capstor/cache/hf_hub" sbatch download_hf_dataset.slurm ocrvqa

# Tune for unstable network
MAX_RETRIES=20 BACKOFF_FACTOR=1.5 sbatch download_hf_dataset.slurm ocrvqa
```

---

### 3. `hf_hub_cache_check.py`

Verifies integrity of downloaded HuggingFace cache files using parallel SHA256 checksum verification.
In thr HF Hub cache files are stored with their sha256 hash as filename. This script verifies the integrity. 
Can be used to make sure source files are valid before processing them. Especially on huge datasets and files on distributed filesystems
this is useful.

**What it does:**
- Recursively scans cache for all blob files (including nested dataset subsets)
- Verifies each blob's SHA256 hash matches its filename
- Uses memory-mapped I/O for efficient processing of large files
- Parallel verification with configurable workers and batching
- Identifies corrupted or mismatched files

**Key options:**
- `--cache-dir`: HF cache directory (default: ~/.cache/huggingface)
- `--dataset`: Verify specific dataset only (e.g., 'username/dataset-name')
- `--workers`: Number of parallel workers (default: CPU count)
- `--batch-size`: Files per batch for granular progress (default: 10)
- `--list`: List all datasets in cache and exit

**Useful Tips:**
- The script prints the corrupted files so you can delete them afterwards manually. 
- When rerun the download script, it will notice the blobs are missing and redownload them.
- HF hub checks file hashes on download, but it doesn't guarantee that the downloaded files remain valid after.
- The hash is also not checked by the hf datasets library before processing (no option to activate this which is a missing feature IMO)

**Example:**
```bash
# Verify all datasets
python hf_hub_cache_check.py

# Verify specific dataset
python hf_hub_cache_check.py --dataset "HuggingFaceM4/FineVision"

# List cached datasets, not run any verification
python hf_hub_cache_check.py --list

# Fast verification with more workers
python hf_hub_cache_check.py --workers 32 --batch-size 20
```

---

### 4. `hf_hub_cache_check.slurm`

SLURM wrapper for `hf_hub_cache_check.py` for cluster-based verification.

**What it does:**
- Runs cache verification as a SLURM job
- Configures parallel workers and batch size
- Provides job logging and completion tracking

**Configuration:**
- `DATASET_NAME`: Specific dataset to verify (default: mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M)
- Workers and batch size are hardcoded in the script (128 workers, batch size 10)

**Example:**
```bash
# Verify default dataset
sbatch hf_hub_cache_check.slurm

# Verify specific dataset
DATASET_NAME="google/docci" sbatch hf_hub_cache_check.slurm
```

---

## Important Notes

**HuggingFace Authentication:**

For private datasets or higher API rate limits, set your HuggingFace token:
```bash
export HF_TOKEN="your_token_here"
```

**Cache Location:**

Make sure you point the cache-dir to the spot where the dataset should be for further processing (dedup, tokenize etc...)

## Typical Workflow

1. **Download dataset:**
   ```bash
   DATASET_NAME="google/docci" sbatch download_hf_dataset.slurm
   ```

2. **Verify integrity if download failed or you want to reprocess a large dataset after some time:**
   ```bash
   DATASET_NAME="google/docci" sbatch hf_hub_cache_check.slurm
   ```

3. **Use in training:**
   Point your training script to the cache directory with `--cache-dir` flag.