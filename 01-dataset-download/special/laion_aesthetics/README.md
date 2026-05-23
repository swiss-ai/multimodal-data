# LAION Aesthetics 12M Download

The `dclure/laion-aesthetics-12m-umap` HuggingFace dataset contains image URLs and
LAION aesthetics scores, but not the images themselves. Downloading requires two steps:

1. **Export URLs** from the HF dataset
2. **Download images** with `img2dataset`

## Step 1 — Export URLs

```bash
pip install datasets

python export_urls.py \
    --output-file all_urls.parquet
```

By default this reads from the HF Hub cache. Set `HF_HUB_CACHE` if you have a shared
cache location.

## Step 2 — Download images with img2dataset

```bash
pip install img2dataset

img2dataset \
    --url_list all_urls.parquet \
    --output_folder /path/to/output \
    --processes_count 64 \
    --thread_count 256 \
    --input_format parquet \
    --url_col url \
    --output_format webdataset \
    --resize_mode no \
    --min_image_size 256 \
    --retries 3 \
    --timeout 30 \
    --number_sample_per_shard 10000
```

Adjust `--processes_count` and `--thread_count` to your machine.
