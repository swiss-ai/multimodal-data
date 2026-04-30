# NVIDIA Llama-Nemotron-VLM-Dataset-v1

Source: https://huggingface.co/datasets/nvidia/Llama-Nemotron-VLM-Dataset-v1

Composition: 21 sub-datasets, 2,863,854 samples total, ~747 GB if images are
all downloaded (the repo itself is ~113 GB; the rest lives in source image
corpora that must be fetched separately).

## What we download

Under a "PD or permissive, no share-alike" licence rule, we keep 17 of 21
sub-datasets:

| # | Sub-dataset   | Category   | Samples   | Size       | Source            | Notes |
|---|---------------|------------|-----------|------------|-------------------|-------|
|  1 | ocr_1        | OCR        | 14,533    | 5.76 GB    | synthetic ASCII   | images included |
|  2 | ocr_2        | OCR        | 29,108    | 15.02 GB   | synthetic words   | images included |
|  3 | ocr_3        | OCR        | 14,525    | 5.65 GB    | synthetic Chinese | images included |
|  4 | ocr_6        | OCR        | 48,369    | 18.59 GB   | DocLayNet         | annotations only; see ocr_6.md for DocLay attribution |
|  5 | ocr_8        | OCR        | 57,137    | 9.30 GB    | FinTabNet         | annotations only |
|  6 | ocr_9        | OCR        | 224,170   | 30.03 GB   | PubTables-1M      | annotations only |
|  7 | ocr_10       | OCR        | 19,379    | 12.92 GB   | Digital Corpora   | annotations only |
|  8 | vqa_1        | VQA        | 1,278,221 | 378.17 GB  | OpenImages        | annotations only |
|  9 | vqa_2        | VQA        | 503,275   | 147.60 GB  | OpenImages        | annotations only |
| 10 | vqa_3        | VQA        | 34,602    | 9.08 GB    | TextVQA           | annotations only |
| 11 | vqa_4        | VQA        | 23,571    | 1.04 GB    | ChartQA           | ⚠ benchmark — see below |
| 12 | vqa_5        | VQA        | 971       | 0.52 GB    | SROIE             | annotations only |
| 13 | vqa_6        | VQA        | 199       | 0.02 GB    | FUNSD             | annotations only |
| 14 | vqa_7        | VQA        | 15,121    | 0.66 GB    | ChartQA           | ⚠ benchmark |
| 15 | vqa_8        | VQA        | 15,050    | 0.64 GB    | ChartQA           | ⚠ benchmark |
| 16 | captioning_1 | Captioning | 21,953    | 5.76 GB    | TextVQA           | annotations only |
| 17 | captioning_2 | Captioning | 109,765   | 28.80 GB   | TextVQA           | annotations only |

Kept samples: 2,409,349 / 2,863,854 (84%).

## What we drop (CC-BY-SA — excluded by licence rule)

| Sub-dataset | Samples   | Size       | Source                |
|-------------|-----------|------------|-----------------------|
| ocr_4       | 188,569   | 32.60 GB   | Rendered Wikipedia    |
| ocr_5       | 193,310   | 32.39 GB   | Rendered Wikipedia    |
| ocr_7       | 25,281    | 2.46 GB    | TabRecSet             |
| vqa_9       | 46,745    | 10.85 GB   | Open textbooks        |

## Contamination check against our existing datasets

None. NVIDIA re-annotates OpenImages / TextVQA / DocLayNet / FinTabNet /
PubTables-1M / Digital Corpora / ChartQA / SROIE / FUNSD — none of these
overlap with OWID, Smithsonian, NASA, or any of our audio sources.

**⚠ ChartQA caveat**: vqa_4/vqa_7/vqa_8 use ChartQA training images. If
ChartQA is used as an evaluation benchmark in Apertus, filter these three
sub-datasets out separately to preserve the eval.

## Source-image downloads still needed

For re-annotated datasets (all rows marked "annotations only" above),
only the JSONL lands locally. Actual images must be obtained from the
original corpus:

- OpenImages → https://storage.googleapis.com/openimages/web/index.html
- TextVQA    → https://textvqa.org/dataset/
- DocLayNet  → https://github.com/DS4SD/DocLayNet
- FinTabNet  → https://developer.ibm.com/data/fintabnet/
- PubTables-1M → https://github.com/microsoft/table-transformer
- Digital Corpora PDFs → https://digitalcorpora.org/
- ChartQA    → https://github.com/vis-nlp/ChartQA
- SROIE / FUNSD → linked in their respective .md files

Run `cat ocr_6.md`, `cat vqa_1.md`, etc. for per-dataset instructions.

## Run

```
sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/llama-nemotron-vlm-v1/download.slurm
```

Output lives at:
`/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/nvidia_nemotron_vlm_v1/`
