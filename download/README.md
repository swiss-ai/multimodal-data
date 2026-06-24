# Dataset Download Scripts

Tools for downloading all datasets used in Apertus training onto the CSCS
filesystems, plus utilities for HuggingFace cache integrity verification,
dataset loading tests, and SHAR repacking.

Most datasets are distributed on HuggingFace Hub and can be downloaded with the
generic scripts in this directory. A few require custom procedures (direct
wget, Zenodo, Kaggle, img2dataset) and live under `special/`. Per-dataset
download recipes (pinned by commit) live under `../datasets/<dataset>/download.slurm`.

## Pick a download recipe by file *shape*

The bottleneck is **how a dataset's bytes are packed**, not its total size:

- **Few large files** (parquet, tar, big shards) → `huggingface-cli` +
  `HF_HUB_ENABLE_HF_TRANSFER=1`, which chunks each big file into parallel byte
  ranges. Template: `../datasets/chartnet_realworldchart/download.slurm`.
- **Many small files** (wav, opus, flac, jpg) → `git clone` + `git lfs pull`.
  `huggingface-cli` pays per-file HTTP overhead, so thousands of tiny files
  crawl; git batches the LFS resolution instead. Template:
  `../datasets/police_scanner/download.slurm`.

## Convention: one `download.slurm` per dataset

Each dataset lives in its own directory under `../datasets/<dataset>/` and
ships a self-contained download script. All download scripts use
**`huggingface-cli download`** (raw files straight to a `--local-dir`, pinned
by `--revision`), not the `datasets` builder. The shape is the same everywhere
— see any recent script such as `../datasets/chartnet_realworldchart/download.slurm`
as the reference template.

Required settings (every HF download script):

- `#SBATCH --account=infra01 --partition=normal --time=12:00:00 --cpus-per-task=288`
- `#SBATCH --environment=nemo_26_02` (verify the current container — rolls periodically)
- `#SBATCH --reservation=SD-69241-apertus-1-5-0` (verify with `scontrol show reservation` — names roll monthly)
- Pin the commit: `--revision <sha>`
- `export HF_HUB_ENABLE_HF_TRANSFER=1` and `--max-workers 64`
- Shared pip packages: set both `PYTHONPATH` and `PATH` to
  `/capstor/store/cscs/swissai/infra01/MLLM/pip-packages`
- Destination naming:
  `/capstor/store/cscs/swissai/infra01/{audio-,vision-}datasets/raw/{hf|ms}___{Provider}___{Dataset}`
- Clean up the hub cache after download: `rm -rf "${HF_HUB_CACHE}"`
- Logs: `/iopsstor/scratch/cscs/%u/apertus/multimodal-data/download/logs/`

Downloading a subset of a repo (e.g. selected languages): pass **all** glob
patterns to a **single** `--include` flag — multiple `--include` flags silently
overwrite each other.

```bash
huggingface-cli download facebook/multilingual_librispeech \
    --repo-type dataset --revision <sha> --local-dir "$DEST" \
    --include "german/*" "dutch/*" "french/*" --max-workers 64
```

Multi-TB datasets should add an `afterany` auto-resubmit loop (one 12 h window
is not enough); see `../datasets/peoples_speech/unsupervised/download.sh` for
the pattern.

`reproduce_apertus_downloads.sh` records the exact commands used for the
Apertus 1.5 download set.

## Quick Start (generic `download_hf_dataset.slurm`)

```bash
# Set cache locations (adjust to your storage)
export HF_HUB_CACHE=/shared/hf_hub_cache
export DATA_ROOT=/shared/vision-datasets

# Optional: authenticate for gated datasets (e.g. google/scin)
huggingface-cli login

bash reproduce_apertus_downloads.sh
```

For datasets that require special handling, see the `special/` directory.

## Dependencies

All required Python packages are specified in `requirements.txt` with minimum
version requirements:

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `huggingface_hub>=0.16.0,<1.0.0` — **CRITICAL**: must be <1.0.0
  (`configure_http_backend` was removed in 1.0.0)
- `datasets>=2.14.0` — dataset loading and preparation
- `hf_transfer>=0.1.0` — optional: faster downloads
- `requests>=2.28.0` — HTTP session with security fixes
- `urllib3>=1.26.0` — retry mechanism
- `tqdm>=4.64.0` — progress bars for cache verification

The SLURM scripts install these dependencies at runtime.

## Scripts Overview

### 1. `download_hf_dataset.py`

Downloads and processes HuggingFace datasets to a local cache with HTTP retry
logic and automatic configuration detection. Handles HF Hub API rate limits via
its retry logic; on failure you can just restart the script and it resumes,
skipping already-downloaded configs.

**What it does:**
- Downloads datasets from HuggingFace Hub with configurable retry logic
- If subset not specified, auto-detects and downloads all dataset configurations
- Skips already-cached configurations automatically
- Supports parallel downloads with configurable workers
- Provides detailed error reports for failed downloads

**Key options:**
- `--dataset-name`: HuggingFace dataset repository (required)
- `--subset-name`: specific config(s) to download (comma-separated), or auto-detect all if omitted
- `--cache-dir`: local HF *datasets* cache (required) — where the processed dataset files end up. Don't confuse with the HF *hub* cache.
- `--num-proc`: parallel download processes (default: auto-detect)
- `--max-retries`: max retry attempts on failure (default: 5)
- `--backoff-factor`: exponential backoff multiplier (default: 1.0)
- `--force-redownload`: force re-download even if cached
- `--trust-remote-code`: allow using dataset script

**Cache Configuration:**

The download process uses **two separate cache locations**:

1. **HF Hub Cache (`HF_HUB_CACHE`)** — raw files downloaded from HuggingFace Hub.
   Default: `~/.cache/huggingface/hub`.
2. **Datasets Cache (`CACHE_DIR`)** — processed datasets ready for use.
   Must be specified via `--cache-dir`.

**On shared clusters:** point both `HF_HUB_CACHE` and `CACHE_DIR` at a shared
filesystem so teams don't re-download the same files. Example:
```
export CACHE_DIR=/shared/vision-datasets/hf_datasets_cache
export HF_HUB_CACHE=/shared/vision-datasets/hf_hub_cache
```

**Tips:**
- For unstable networks, raise `--max-retries` and `--backoff-factor`.
- HF datasets are first downloaded by the Hub, then processed by the
  `datasets` library. If processing errors out after a clean download, the
  cache may be corrupted (common on distributed FS with large datasets) —
  run `hf_hub_cache_check.py` to detect and re-pull missing/corrupted blobs.
- The retry logic respects HF's `Retry-After` headers; you will not be blocked.

**Example:**
```bash
# Download all configs
python download_hf_dataset.py \
    --dataset-name "HuggingFaceM4/FineVision" \
    --cache-dir "/path/to/cache"

# Specific configs with retry tuning
python download_hf_dataset.py \
    --dataset-name "ibm-research/duorc" \
    --subset-name "ParaphraseRC,SelfRC" \
    --cache-dir "./cache" \
    --max-retries 10 \
    --backoff-factor 1.5
```

### 2. `download_hf_dataset.slurm`

SLURM wrapper for `download_hf_dataset.py` with environment setup and job
management. Installs deps from `requirements.txt`, cleans stale lock files,
and configures resources.

**Configuration via environment variables:**
- `DATASET_NAME`, `SUBSET_NAME`, `CACHE_DIR`, `HF_HUB_CACHE`, `NUM_PROC`,
  `MAX_RETRIES`, `BACKOFF_FACTOR`, `FORCE_REDOWNLOAD`, `USE_HF_TRANSFER`,
  `CLUSTER_REPO_HOME`.
- `CLUSTER_REPO_HOME` defaults to `SLURM_SUBMIT_DIR` and must point to the
  repository checkout in the cluster environment.

**Example:**
```bash
sbatch download_hf_dataset.slurm                                  # defaults
DATASET_NAME="google/docci" sbatch download_hf_dataset.slurm ""   # auto-detect all configs
HF_HUB_CACHE="/shared/hf_hub_cache" sbatch download_hf_dataset.slurm ocrvqa
MAX_RETRIES=20 BACKOFF_FACTOR=1.5 sbatch download_hf_dataset.slurm ocrvqa
```

### 3. `hf_hub_cache_check.py` / `hf_hub_cache_check.slurm`

Parallel SHA256 verification of HuggingFace cache blobs (filename *is* the
sha256). Useful for catching corruption on the distributed filesystem after a
large download, before processing. The SLURM wrapper runs the same check as a
cluster job.

```bash
python hf_hub_cache_check.py                                    # all datasets
python hf_hub_cache_check.py --dataset "HuggingFaceM4/FineVision" --workers 32
python hf_hub_cache_check.py --list                             # list cached datasets, no verify
sbatch hf_hub_cache_check.slurm
DATASET_NAME="google/docci" sbatch hf_hub_cache_check.slurm
```

**Key options (Python):** `--cache-dir`, `--dataset`, `--workers`,
`--batch-size`, `--list`.

Corrupted blobs are printed so you can delete them; re-running the download
re-fetches the missing blobs. HF Hub verifies hashes only at download time —
this script lets you re-verify after the fact.

### 4. `load_dataset_test.py` / `load_dataset_test.slurm`

Tests loading of downloaded HuggingFace datasets with comprehensive statistics
reporting.

**Methods:** `"default"` (`load_dataset`) or `"builder_load"`
(`builder.as_dataset`). Auto-detects all configs when `--subset-name` is
omitted; supports comma-separated config names for batch testing. Prints
condensed stats for multi-config tests, full details for single-config.
Supports streaming and split slicing (`"train[:100]"`).

**Key options:** `--dataset-name` (required), `--subset-name`, `--split`,
`--cache-dir` (required), `--method`, `--streaming`, `--num-proc`.

**SLURM env vars:** `DATASET_NAME`, `SUBSET_NAME`, `SPLIT`, `CACHE_DIR`,
`HF_HUB_CACHE`, `METHOD`, `STREAMING`, `NUM_PROC`, `CLUSTER_REPO_HOME`.

```bash
# Test all configs (auto-detect)
python load_dataset_test.py \
    --dataset-name "ibm-research/duorc" \
    --cache-dir "~/.cache/huggingface/datasets"

# Single config, full details, builder_load
python load_dataset_test.py \
    --dataset-name "google/docci" --subset-name "default" \
    --cache-dir "./cache" --method builder_load

# Streaming + slice
python load_dataset_test.py \
    --dataset-name "google/docci" --cache-dir "./cache" \
    --method default --streaming --split "train[:1000]"

# SLURM
sbatch load_dataset_test.slurm
DATASET_NAME="google/docci" sbatch load_dataset_test.slurm
```

### 5. `merge_shar.py`

Merges existing Lhotse SHAR shards into fewer, larger shards **without**
re-decoding/resampling/re-tokenizing. Preserves field order and alignment from
`shar_index.json` so `CutSet.from_shar(...)` stays valid.

### 6. `rsync_*_to_capstor.slurm`

SLURM jobs that rsync staged datasets between scratch and capstor storage:

- `rsync_audio_only_to_capstor.slurm` — push audio-dataset-only payloads.
- `rsync_ups_to_capstor.slurm` — generic upstream-to-capstor push.
- `rsync_voxpopuli_to_capstor.slurm` — VoxPopuli-specific transfer.

### 7. `to_shar_examples.sh`

Reference commands for converting various source formats into Lhotse SHAR
shards. Use as a cookbook when adding a new dataset's `prepare_to_shar.slurm`.

## `special/` — Dataset-specific Download Scripts

Some datasets cannot be retrieved with a plain `hf download` and need extra
steps.

| Directory | Dataset | Method |
|---|---|---|
| `special/swissimage/` | SWISSIMAGE | WMS tile download via swisstopo API + img2dataset |
| `special/laion_aesthetics/` | LAION Aesthetics 12M | Export URLs from HF + img2dataset |
| `special/pixmo_cap/` | PixMo-Cap images | Export URLs from HF + img2dataset |
| `special/medmax/` | MedMax | wget split archives from HF resolve |
| `special/medmnist/` | MedMNIST | Zenodo download via zenodo_get |
| `special/slide/` | SLIDE | wget from Figshare |
| `special/skyscript/` | SkyScript | wget from AWS S3 |
| `special/bigdocs/` | BigDocs-7.5M (extra) | wget external image archives (COCO, TextVQA, TableFact) |
| `special/medical/` | Various medical datasets | Kaggle CLI + wget from Zenodo/Figshare |

## `audio/` — Cross-dataset audio helpers

Pipeline-stage scripts that operate across audio datasets:

- `build_all_interleave.slurm` — build interleaved audio+text shards across
  all configured datasets.
- `build_voxpopuli_interleave.slurm` — VoxPopuli-specific interleave builder.
- `run_vad_multichannel.py` — multichannel voice-activity detection helper
  used by several speech datasets.

## Authentication

For gated/private datasets or higher rate limits:

```bash
export HF_TOKEN="$(cat $HOME/.hf-token)"
# or
export HF_TOKEN="your_token_here"
```

## Typical Workflow

1. **Per-dataset download** (most common):
   `sbatch <abs-path>/datasets/<dataset>/download.slurm`
2. **Generic HF download**:
   `DATASET_NAME="google/docci" sbatch download_hf_dataset.slurm`
3. **Verify integrity** (failed download or large dataset re-process):
   `DATASET_NAME="google/docci" sbatch hf_hub_cache_check.slurm`
4. **Test loading**:
   `DATASET_NAME="google/docci" sbatch load_dataset_test.slurm`
5. **Process:** hand the `raw/<...>` dir to the dataset's
   convert/tokenize/SHAR steps.
