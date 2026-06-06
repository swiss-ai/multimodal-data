# Dataset Download Scripts

Tooling for downloading multimodal (audio + image) datasets onto the CSCS
filesystems for Apertus training, plus helpers for integrity verification and
SHAR repacking.

## Pick a download recipe by file *shape*

The bottleneck is **how a dataset's bytes are packed**, not its total size:

- **Few large files** (parquet, tar, big shards) → `huggingface-cli` +
  `HF_HUB_ENABLE_HF_TRANSFER=1`, which chunks each big file into parallel byte
  ranges. Template:
  [`image/chartnet_realworldchart/download.slurm`](image/chartnet_realworldchart/download.slurm).
- **Many small files** (wav, opus, flac, jpg) → `git clone` + `git lfs pull`.
  `huggingface-cli` pays per-file HTTP overhead, so thousands of tiny files
  crawl; git batches the LFS resolution instead. Template:
  [`audio/police_scanner/download.slurm`](audio/police_scanner/download.slurm).

## Convention: one `download.slurm` per dataset

Each dataset lives in its own directory and ships a self-contained download
script:

```
01-dataset-download/
  audio/<dataset>/download.slurm
  image/<dataset>/download.slurm
```

All download scripts use **`huggingface-cli download`** (raw files straight to a
`--local-dir`, pinned by `--revision`), not the `datasets` builder. The shape is
the same everywhere — see any recent script such as
`image/chartnet_realworldchart/download.slurm` as the reference template.

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
- Logs: `/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/`

Downloading a subset of a repo (e.g. selected languages): pass **all** glob
patterns to a **single** `--include` flag — multiple `--include` flags silently
overwrite each other.

```bash
huggingface-cli download facebook/multilingual_librispeech \
    --repo-type dataset --revision <sha> --local-dir "$DEST" \
    --include "german/*" "dutch/*" "french/*" --max-workers 64
```

Multi-TB datasets should add an `afterany` auto-resubmit loop (one 12 h window is
not enough); see `audio/peoples_speech/unsupervised/download.sh` for the pattern.

`reproduce_apertus_downloads.sh` records the exact commands used for the Apertus
1.5 download set.

## Shared helpers

### `hf_hub_cache_check.py` / `.slurm`

Parallel SHA256 verification of HuggingFace cache blobs (filename *is* the
sha256). Useful for catching corruption on the distributed filesystem after a
large download, before processing.

```bash
# Verify all datasets in the cache
python hf_hub_cache_check.py

# Verify one dataset; tune workers
python hf_hub_cache_check.py --dataset "HuggingFaceM4/FineVision" --workers 32

# List cached datasets without verifying
python hf_hub_cache_check.py --list
```

Corrupted blobs are printed so you can delete them; re-running the download
re-fetches the missing blobs.

### `merge_shar.py`

Merges existing Lhotse SHAR shards into fewer, larger shards **without**
re-decoding/resampling/re-tokenizing. Preserves field order and alignment from
`shar_index.json` so `CutSet.from_shar(...)` stays valid.

## Authentication

For gated/private datasets or higher rate limits:

```bash
export HF_TOKEN="$(cat $HOME/.hf-token)"
```

## Typical workflow

1. **Download** — `sbatch <abs-path>/audio_or_image/<dataset>/download.slurm`
2. **Verify (optional, large datasets)** — `python hf_hub_cache_check.py --dataset "<repo>"`
3. **Process** — hand the `raw/<...>` dir to the dataset's convert/tokenize/SHAR steps.
