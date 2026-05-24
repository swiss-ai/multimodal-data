# SWISSIMAGE Download

SWISSIMAGE is the Swiss Federal Office of Topography's (swisstopo) high-resolution
aerial orthophoto mosaic. It is available for free via a WMS (Web Map Service) API
at `https://wms.geo.admin.ch/`.

Downloading requires two steps:

1. **Generate a URL list** from a tile grid covering the desired area
2. **Download tiles** with `img2dataset`

## Step 1 — Generate URLs

The tile grid is a CSV with `bbox` columns in LV-95 (EPSG:2056) coordinates.
`generate_urls.py` builds WMS GetMap URLs for each tile:

```bash
python generate_urls.py \
    --input-csv tiles.csv \
    --output-csv swissimage_urls.csv \
    --width 1024 \
    --height 1024
```

The tile grid CSV must have a `bbox` column in the format `minX,minY,maxX,maxY`.
Tiles can be generated with any GIS tool or downloaded from swisstopo's geodata
portal (https://www.swisstopo.admin.ch/en/geodata).

## Step 2 — Download tiles

```bash
pip install img2dataset

img2dataset \
    --url_list swissimage_urls.csv \
    --output_folder /path/to/swissimage \
    --processes_count 32 \
    --resize_mode no \
    --output_format webdataset \
    --input_format csv \
    --url_col url \
    --number_sample_per_shard 100 \
    --timeout 120 \
    --retries 5
```

## License

SWISSIMAGE is available under the Open Government Data (OGD) licence.
See https://www.swisstopo.admin.ch/en/terms-of-use-free-geodata-services for details.
