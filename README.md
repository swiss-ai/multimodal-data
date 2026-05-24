# Apertus Data Preparation

Complete data preprocessing toolkit for Apertus multimodal VLM training.
Covers data download, deduplication, format conversion, recaptioning with
open-weight models, image-text pairing, and verification across 60+ datasets.

## Directory Structure

```
.
├── README.md
├── medical/           # Medical data pipeline (streaming adapters + filters + writers)
├── download/          # Dataset download scripts (HF Hub, S3, img2dataset, special)
├── dedup/             # Perceptual-hash deduplication pipeline (hash→classify→rewrite)
├── recaption/         # Recaption engines (vLLM Qwen3.5, BLIP3, structured SFT)
├── convert/           # Image format conversion (PNG→JPEG, TIFF→JPEG, text cleaning)
├── image_text/        # Image-text pairing utilities (JSON, parquet, ZIP sources)
├── datasets/          # Per-dataset preprocessing scripts
│   ├── bigdocs/       # BigDocs-7.5M parquet/zips → webdataset
│   ├── bigearthnet/   # BigEarthNet Sentinel-2 bands → RGB + QA conversations
│   ├── ccpdf/         # CommonCanvas PDF download + parquet
│   ├── dailymed/      # DailyMed SPL nested zips → parquet
│   ├── ego4d/         # Ego4D video frame extraction + narration pairing
│   ├── egopat3dv2/    # EgoPAT3Dv2 HDF5 → triplet frame webdataset
│   ├── facetaption/   # FaceCaption-15M bounding-box face cropping
│   ├── flair_hub/     # FLAIR-HUB TIFF zips → JPEG webdataset
│   ├── laion/         # LAION-Aesthetics-12M-UMAP preprocessing
│   ├── mint_arxiv/    # MINT-1T-ArXiv deduplication + TIFF processing
│   ├── mint_html/     # MINT-1T-HTML deduplication
│   ├── nasa/          # NASA image parquet export
│   ├── nemotron/      # Nemotron dataset format conversion
│   ├── openimages/    # OpenImages V7 download + webdataset packing
│   ├── skyscript/     # SkyScript remote sensing webdataset processing
│   ├── smithsonian/   # Smithsonian Open Access 5-phase pipeline
│   └── swisstopo/     # Swisstopo map tile download + captioning
└── tools/             # Utility scripts (counting, verification, extraction, stats)
```

## Datasets Covered

### Stage 1 — Image-Only Pretraining
Medical (BUSI, COVIDx-US, DDTI, ISIC, MedMax, MedMNIST, MedTrinity, OPEN-PMC,
PMC-OA, RFMiD2, SCIN, SLIDE, UWF, HoloAssist), SWISSIMAGE,
LAION-Aesthetics-12M-UMAP, MINT-1T (HTML, PDF, ArXiv), BigDocs,
Copernicus-Bench, TreeOfLife-10M

### Stage 2 — Image+Text
Recap-DataComp-1B, LLaVA-OneVision, CommonCatalog-CC-BY, PD12M,
BLIP3-Grounding-50M, PIN-200M, BLIP3o-Pretrain, MIT-10M, Art-Museums-PD,
Megalith-10M, WebSight, Open-PMC, TextAtlas5M, OpenImages-V7, MedTrinity-25M,
UNO-1M, FaceCaption-15M, PixMo-Cap, Molmo2-SynMultiImageQA, SkyScript,
TCM-Pretrain, Fine-T2I, MedMax, RSTeller, MapTrace, PMC-OA, GeoChat,
NIH-Chest-Xray, DaTikZ, FLAIR-HUB, Product-Catalogue, MultiCaRe,
NCT-CRC-HE-100K, Diabetic-Retinopathy, Brain-Tumor-MRI, Liver-Ultrasound,
Breast-Ultrasound, EBHI-Seg, MedPix-2

### Cooldown Datasets
DOCCI, OWID, WAFFLE, HQ-50K, Smithsonian Open Access, DailyMed SPL, NASA,
Swisstopo Maps, MINT-1T-ArXiv, ChartNet, PixMo-Point-Explanations,
LaTeX-Formulas, Visual Genome, CreLLO

### Long Context
TCM, MedPix-2, PIN-200M, Argimi-Finance, Molmo2-SynMultiImageQA,
Art-Museums-PD, Swisstopo Map-Sat, OWID, HQ-50K, NASA, MINT-1T-ArXiv,
ChartNet, WAFFLE

### SFT / Instruction Tuning
Innovator-VL-Instruct, LLaVA-OneVision-Instruct, SenseNova-SI-8M,
Nemotron-Image-Training-v3, MapTrace, LNQA, PixMo-Cap-QA, CoSyn-400K,
BigEarthNet, RSRCC, PixMo-Ask-Model-Anything, MMEvol, TCM-Instruction,
Molmo2-SynMultiImageQA, MolmoPoint-GUISyn, VDR-Cooking, ChineseMeme,
Molmo2-MultiImageQA, Path-VQA, MultiHiertt, Memotion, SpatialSense

## Quick Start

### Prerequisites

```bash
# Core dependencies
pip install webdataset opencv-python pillow pyarrow pandas rocksdict tqdm

# HuggingFace Hub
pip install huggingface_hub datasets

# Specific dataset dependencies (install as needed)
pip install h5py          # EgoPAT3Dv2
pip install rasterio      # BigEarthNet
pip install vllm          # Recaption engine
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace API token (required for gated datasets) |
| `HF_HOME` | HuggingFace cache directory |
| `HF_HUB_CACHE` | Hub snapshot cache directory |
| `DATA_ROOT` | Root directory for dataset storage |

### Common Workflows

**Deduplicate a webdataset (3 stages):**
```bash
# Stage 1: Hash
python dedup/hash_webdataset.py --input "data/*.tar" --output-dir hashes/

# Stage 2: Classify
python dedup/classify_hashes.py --hash-dir hashes/ --db-path dedup.db --reject-list rejects.txt

# Stage 3: Rewrite
python dedup/rewrite_clean.py --input "data/*.tar" --output clean/ --reject-list rejects.txt
```

**Convert PNG to JPEG:**
```bash
python convert/png_to_jpeg.py data/ output/ --quality 95 --resize 512 512
```

**Run medical pipeline:**
```bash
cd medical && python main.py configs/apertus_image_only_v2.json
```

**Pair JSON annotations with images:**
```bash
python image_text/rsteller.py  # edit paths in script first
```

**Recaption with vLLM (SLURM array job):**
```bash
RECAPTION_LOADER=loader_recap_datacomp_1b_downloaded_v2 \
  sbatch --array=0-99 recaption/vllm/run.slurm
```

## Medical Pipeline (`medical/`)

A fault-tolerant streaming pipeline for loading, filtering, and writing large
medical multimodal datasets. See `medical/README.md` for details.

### Adapters (24 datasets)
BUSI, COVIDx-US, DDTI, ISIC, MedMax, MedMNIST, MedTrinity, OPEN-PMC-18M, PMC-OA,
RFMiD2, SCIN, SLIDE, UWF Fundus, HoloAssist, MediCaT, MMC4, MultiCaRe,
Brain-Tumor-MRI, Diabetic-Retinopathy, COVID-Radiography, EBHI-Seg,
Liver-Ultrasound, NCT-CRC-HE-100K, NIH-Chest-Xray

### Filters
- Image resolution filtering (min/max dimensions)
- Perceptual hash deduplication (LMDB-backed)
- Downsampling for large images

### Writers
- HuggingFace datasets (sharded parquet)
- Webdataset (tar shards)

## Recaption Pipelines (`recaption/`)

### vLLM Engine (`recaption/vllm/`)
Streaming batch recaption using Qwen3.5-9B (or any vLLM-supported VLM).
17 dataset loaders included for: CommonCatalog, MIT-10M, Mitsua Art Museums,
OpenImages V7, Recap-DataComp-1B, Spawning PD12M, UNO-1M, WAFFLE.

Each loader defines: data source, sampling strategy, prompt template, and
model configuration. New loaders follow the pattern in `recaption/vllm/loaders/`.

### BLIP3 (`recaption/blip3/`)
Multi-GPU BLIP3 recaption pipeline with chunk splitting, merge, and validation.

### Structured SFT (`recaption/sft/`)
Two-stage framework for high-quality instruction data generation:
1. **Generation**: VLM produces reasoning QA pairs from images
2. **Judging**: Second VLM evaluates groundedness, specificity, and quality

Loaders for: HQ-50K, Google RSRCC, DRiM-VisualReason-Hard, MINT-1T-ArXiv.

## Deduplication Pipeline (`dedup/`)

### Three-Stage Architecture

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `hash_webdataset.py` | Compute pHash for all images, write to parquet |
| 2 | `classify_hashes.py` | Check against RocksDB, generate reject list |
| 3 | `rewrite_clean.py` | Drop duplicates, write clean shards |

### Dataset-Specific Variants

| Script | Dataset | Notes |
|--------|---------|-------|
| `dedup_mint_html.py` | MINT-1T-HTML | Large-scale 3-stage with sharded DB |
| `dedup_mint_arxiv.py` | MINT-1T-ArXiv | HF `datasets` streaming |
| `dedup_mint_pdf.py` | MINT-1T-PDF | Multi-page TIFF splitting |
| `dedup_bigdocs.py` | BigDocs-7.5M | Parquet-based dedup |
| `dedup_swisstopo.py` | Swisstopo Maps | WDS-based with URL keys |
| `dedup_wds.py` | Generic WDS | Master script with configurable stages |

## Dataset-Specific Scripts (`datasets/`)

### BigDocs (`datasets/bigdocs/`)
Converts BigDocs-7.5M subsets (ArxivOCR, ArxivTableCap, cord-v2, pubtables-1m)
and external sources (COCO-Text, TextOCR) from parquet/zips to webdataset.

### BigEarthNet (`datasets/bigearthnet/`)
Converts Sentinel-2 multispectral bands (B02, B03, B04) to RGB PNG and pairs
with QA conversations from BigEarthNet.txt parquet. Requires rasterio and pzstd.

### EgoPAT3Dv2 (`datasets/egopat3dv2/`)
Reads HDF5 RGB recordings, applies blur detection, near-duplicate hashing,
and temporal sampling. Outputs triplet frames as webdataset shards.

### Smithsonian (`datasets/smithsonian/`)
5-phase pipeline: metadata indexing → caption building → webdataset packing →
caption cleaning (with VLM verification) → resharding. Handles art/history,
NMNH natural history, and 3D render collections (CC0 licensed).

### Swisstopo (`datasets/swisstopo/`)
Swiss national map tile download, building filtering, image captioning with VLM,
and parquet export pipeline.

### DailyMed SPL (`datasets/dailymed/`)
Extracts nested ZIP archives (outer zip → inner zips → XML + images) into
flat parquet shards. 256-way parallel processing.

### FLAIR-HUB (`datasets/flair_hub/`)
Converts GeoTIFF images stored in ZIP archives to RGB JPEG webdataset shards.

### Ego4D (`datasets/ego4d/`)
Frame extraction from full-scale videos using narration timestamps, with
redacted interval filtering and two-step ffmpeg seeking.

### FaceCaption-15M (`datasets/facetaption/`)
Face bounding-box cropping pipeline: reads downloaded images, crops to
face regions, writes cropped images + captions + metadata to tar+parquet.

### Nemotron (`datasets/nemotron/`)
Converts Nemotron training datasets (CLEVR, DocVQA, ECD) to standard format.

### SkyScript (`datasets/skyscript/`)
Processes SkyScript remote sensing caption dataset into webdataset format.

## Image-Text Pairing (`image_text/`)

| Script | Source | Output |
|--------|--------|--------|
| `rsteller.py` | RSTeller JSON annotations + JPG | WDS with `<\|img1\|>` captions |
| `geochat.py` | GeoChat JSON + multi-part ZIP | WDS with conversation format |
| `maptrace.py` | MapTrace parquet (image_bytes + caption) | WDS with dedup |
| `geospatial.py` | Geospatial captions dataset | WDS pairing |

## Download (`download/`)

### HF Hub Download
```bash
python download/download_hf_dataset.py --dataset org/dataset --output-dir /data/
```

### img2dataset Pipeline
Three-stage pipeline for datasets that provide URL + caption metadata:
1. `get_metadata.py` — Extract (url, caption) from HF parquet
2. `filter_urls.py` — Robots.txt compliance and URL filtering
3. `img2dataset_download.py` — Parallel image download with img2dataset

### Special Downloads
Dataset-specific download scripts in `download/special/`:
- `swissimage/` — WMS tile URL generation + img2dataset
- `laion_aesthetics/` — URL extraction from HF dataset
- `pixmo_cap/` — URL validation + img2dataset
- `medmax/` — wget from HF resolve
- `medmnist/` — zenodo_get from Zenodo
- `slide/` — wget from Figshare
- `skyscript/` — wget from AWS S3
- `bigdocs/` — External archives (COCO, TextVQA, TableFact)
- `medical/` — Kaggle CLI + wget for Zenodo/Figshare datasets

### Reproduce Full Apertus Downloads
```bash
bash download/reproduce_apertus_downloads.sh
```
Downloads all Stage 1, Stage 2, and SFT datasets with inline notes for
gated and access-restricted datasets.

## Utility Tools (`tools/`)

| Tool | Description |
|------|-------------|
| `count_samples.py` | Count samples in WDS, tar.gz, or parquet files |
| `check_tar_integrity.py` | Detect corrupted/empty tar shards |
| `verify_dataset.py` | Full validation: image decode, UTF-8, key uniqueness |
| `extract_samples.py` | Extract sample images/text for inspection |
| `check_duplicates.py` | Find duplicate images by hash across shards |
| `detect_text_repetition.py` | Detect repetitive text in captions |
| `normalize_bbox.py` | Normalize bounding box coordinates |
| `verify_bbox.py` | Verify bounding box correctness |
| `wds_stats.py` | Compute statistics for webdataset shards |
| `compress_dataset.py` | Compress dataset directories to tar archives |

## Design Principles

1. **Self-contained**: Each script has its own CLI with `--help`. No framework
   dependency required.
2. **Parameterized**: All paths, model names, and thresholds are configurable.
   No hardcoded infrastructure paths in production code.
3. **Resumable**: Pipelines support checkpointing. Interrupted jobs can resume
   without re-processing.
4. **Parallel**: Multiprocessing for CPU-bound tasks (hashing, decoding);
   streaming for I/O-bound tasks (tar reading, network).
5. **Clean**: No credentials, tokens, or internal infrastructure paths in the
   published repository. Set sensitive values via environment variables.

## Notes

- Some datasets require authentication (gated HF repos, Kaggle). See
  individual dataset READMEs for access instructions.
- The SCIN dataset and certain medical datasets require institutional access
  approval before downloading.
- Model weights for recaptioning are not included. Download them separately
  from HuggingFace Hub and set the `MODEL_DIR` or model path accordingly.
