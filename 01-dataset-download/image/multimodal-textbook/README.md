# DAMO-NLP-SG / multimodal_textbook

Source: https://huggingface.co/datasets/DAMO-NLP-SG/multimodal_textbook
Paper:  "2.5 Years in Class: A Multimodal Textbook for Vision-Language
         Pretraining" (ICCV 2025 Highlight, arXiv 2501.00958)

## Content

- **6.5M keyframes** extracted from 75K instructional videos (22,697 hours)
- **0.8B ASR tokens** (259M raw ASR + 500M OCR tokens) aligned to keyframes
- **610K interleaved samples** in OBELICS format
  - avg 10.7 keyframes + 1,230 text tokens per sample
- Subjects: mathematics, physics, chemistry, geography, CS, engineering
- Language: English

## Files

| File | Size | Notes |
|---|---|---|
| `multimodal_textbook.json` | 11.8 GB | All 610K samples |
| `multimodal_textbook_face_v1_th0.04.json` | 11.7 GB | Same, faces-filtered (th=0.04) |
| `dataset_images_interval_7.tar.gz.part_{00..19}` | 20 × 28.3 GB = ~566 GB | Split image archive |
| `video_meta_data/*.json` | small | Video metadata (159K videos) |
| `playground.zip` | 16 GB | **excluded** — demo notebooks, not training data |

After extraction:
```
dataset_images_interval_7/
├── {video_id}/
│   └── {video_id}@{start_sec}_{end_sec}#{kf_num}.jpg
└── ...
```

## License

**Apache 2.0** — allowed under PD-or-permissive-non-SA rule.

## Contamination check against our existing datasets

No overlap with OWID, NASA, Smithsonian, or any of our audio corpora.
Instructional-video keyframes from YouTube education channels — an entirely
different distribution from anything else in the pipeline.

**⚠ Possible overlap with upstream YODAS / YTC**: if you're downloading the
YODAS or YouTube Creative Commons audio dumps, there is a small but non-zero
chance some of the same videos appear in both. The textbook samples are
restricted to explicitly educational channels though, so overlap should be
minimal and can be detected post-hoc via `video_id` columns.

## Disk footprint during download

Peak ~1.2 TB (parts + merged + extracted); settles to ~600 GB after the
SLURM script deletes the tar.gz parts and merged archive.

Final target: 600 GB at
`/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/damo_multimodal_textbook/`

## Run

```
sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/multimodal-textbook/download.slurm
```
