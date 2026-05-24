# PixMo-Cap Image Download

`allenai/pixmo-cap` is a caption dataset that contains image URLs and transcripts,
but not the images themselves. Downloading the actual images requires `img2dataset`.

## Steps

### 1. Download the metadata

```bash
hf download --repo-type dataset allenai/pixmo-cap --local-dir ./pixmo-cap-meta
```

### 2. (Optional) Filter to the desired subset

The metadata parquet files may contain entries without valid image URLs. The
`filter.py` script drops those rows before downloading:

```bash
python filter.py \
    --input-dir ./pixmo-cap-meta \
    --output-file filtered.parquet
```

### 3. Download images with img2dataset

```bash
pip install img2dataset

img2dataset \
    --url_list filtered.parquet \
    --output_folder /path/to/pixmo-cap-images \
    --processes_count 64 \
    --thread_count 64 \
    --input_format parquet \
    --url_col image_url \
    --caption_col caption \
    --save_additional_columns '["transcripts"]' \
    --output_format webdataset \
    --resize_mode no \
    --encode_format png \
    --number_sample_per_shard 10000 \
    --timeout 30 \
    --retries 1
```
