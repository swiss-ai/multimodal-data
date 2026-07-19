# Apertus v1.5 Multimodal Data Preparation

Preprocessing toolkit for Apertus v1.5 multimodal training data. Covers
download, deduplication, format conversion, recaptioning, image-text pairing,
tokenization, and verification across image, image-text, and audio datasets.

The dataset inventory (license, modality, stage, processing scripts, upstream
source) lives in [`datasets/DATASETS.md`](datasets/DATASETS.md). It is generated
from `datasets/inventory.yaml` by `datasets/build_index.py`.

## Layout

| Path | Purpose | Docs |
|------|---------|------|
| `datasets/` | Per-dataset preprocessing scripts | [README](datasets/README.md), [DATASETS.md](datasets/DATASETS.md) |
| `download/` | Dataset download (HF Hub, S3, img2dataset, audio, special) | [README](download/README.md) |
| `medical/` | Streaming adapter pipeline for medical datasets (adapters, filters, writers) | [README](medical/README.md) |
| `preprocessing/` | Earlier adapter framework, predecessor of `medical/`, pending consolidation | [README](preprocessing/README.md) |
| `recaption/` | Recaption engines: vLLM, BLIP3, structured SFT generation and judging | [README](recaption/README.md) |
| `dedup/` | Perceptual-hash deduplication (hash, classify, rewrite stages) | [README](dedup/README.md) |
| `convert/` | Image and text format conversion (PNG and TIFF to JPEG, text cleaning) | [README](convert/README.md) |
| `image_text/` | Image-text pairing (JSON, parquet, or ZIP sources to webdataset) | [README](image_text/README.md) |
| `transcription/` | Audio transcription jobs | [README](transcription/README.md) |
| `tokenization/` | Text-SFT tokenization pipeline | [README](tokenization/README.md) |
| `tools/` | Verification and stats utilities (count, tar integrity, dedup check, bbox) | [README](tools/README.md) |

## Quick start

Every script is self-contained with its own `--help`. Install only what a step needs.

```bash
pip install webdataset opencv-python pillow pyarrow pandas rocksdict tqdm huggingface_hub datasets
```

Common environment variables:
`HF_TOKEN` for gated repos, `HF_HOME` and `HF_HUB_CACHE` for caches, `DATA_ROOT` for dataset storage.

```bash
# Deduplicate a webdataset (3 stages)
python dedup/hash_webdataset.py  --input "data/*.tar" --output-dir hashes/
python dedup/classify_hashes.py  --hash-dir hashes/ --db-path dedup.db --reject-list rejects.txt
python dedup/rewrite_clean.py    --input "data/*.tar" --output clean/ --reject-list rejects.txt

# Convert PNG to JPEG
python convert/png_to_jpeg.py data/ output/ --quality 95 --resize 512 512

# Run the medical pipeline
cd medical && python main.py configs/apertus_image_only_v2.json

# Recaption with vLLM (SLURM array job)
RECAPTION_LOADER=loader_recap_datacomp_1b_downloaded_v2 sbatch --array=0-99 recaption/vllm/run.slurm

# Reproduce the full Apertus download set
bash download/reproduce_apertus_downloads.sh
```

See each subdirectory's README for pipeline-specific details.

## Notes

- No credentials or infrastructure paths are committed. Pass sensitive values through environment variables.
- Some datasets are gated (HF, Kaggle) or need special access such as SCIN. See the per-dataset scripts and READMEs for access instructions.
- Recaption model weights are not included. To recap, download them separately and point the loader or config at them.
