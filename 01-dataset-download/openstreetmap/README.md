# Getting Tiles from OpenStreetMap Data

This directory contains scripts to download and process OpenStreetMap (OSM) data for generating map tiles.

## Overview

1. Download the planet `.osm.pbf` file using `dl_planet.sh`.
2. Slice the planet file into smaller bounding boxes using `slice_osm.sh`.
3. Convert the sliced OSM files into PostgreSQL databases using `import_osm.sh`.
4. Run the tile server and scrape tiles using `run.sh`.

## Requirements

- Podman
    - Or Docker with minor modifications to the scripts
- [Osmium Tool](https://osmcode.org/osmium-tool/)
- [uv](https://github.com/astral-sh/uv)
    - Or other Python environment management tool
- SLURM workload manager for job scheduling
- Sufficient disk space for OSM data and tiles
    - For reference, the planet file is over 80GB, and processed data can require several TBs.

## Downloading OSM Planet File

To download the entire OSM planet file, with:

```plaintext
bash dl_planet.sh --download-dir <path to download directory>
```

This uses [openmaptiles-tools](https://github.com/openmaptiles/openmaptiles-tools) in a Podman container to handle the download.

## Slicing 

Due to memory constraints, the planet file is sliced into smaller bounding boxes. Use:

```plaintext
bash slice_osm.sh \
    --planet-path <path to planet.osm.pbf> \
    --slice-dir <path to output slice directory> \
    --west -180 --east 180 \
    --south -90 --north 90 \
    --lon-step 30 --lat-step 15
```

This will create multiple `slice_<west>_<south>_<east>_<north>.pbf` files in the specified slice directory.

## Importing Slices into PostgreSQL

To import the sliced OSM files into PostgreSQL databases, use:

```plaintext
bash import_osm.sh \
    --slice-dir <path to slice directory> \
    --volume-dir <path to output volume directory>
    --import-split <split index from 0> \
    --total-splits <total number of splits>
```

This will create a `osm-data_<bounding box>.tar` file for each slice in the specified volume directory. The tar files are exported podman volumes containing the PostgreSQL databases.

If you encounter memory issues during import, consider reducing the size of the bounding boxes in the slicing step.

## Running Tile Server and Scraping Tiles

Finally, to run the tile server and scrape tiles, use:

```plaintext
bash run.sh \
    --volume-dir <path to volume directory> \
    --save-dir <path to save tiles> \
    --zoom-level <zoom level> \
    --sample-ratio <sampling ratio from 0 to 1> \
    --run-split <split index from 0> \
    --total-splits <total number of splits>
```

This will start the tile server using the specified volume directory and scrape tiles at the given zoom level, saving them to the specified save directory. The sampling ratio controls the fraction of tiles to download, and the run split parameters allow for parallel execution.

The tiles are saved as `<zoom>_<x>_<y>.png` files in the specified save directory.

For setting the correct zoom levels and understanding tile coordinates, refer to the [Zoom Levels](https://wiki.openstreetmap.org/wiki/Zoom_levels) and [Slippy Map Tilenames](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames) documentation.

## (Optional) Scrape Script

The `run.py` script can also be used independently to scrape tiles from a running tile server. Usage:

```plaintext
run.py [-h] [--bbox MIN_LON MIN_LAT MAX_LON MAX_LAT] --zoom ZOOM [--sample_ratio SAMPLE_RATIO] --save_dir SAVE_DIR [--url URL]

options:
  -h, --help            show this help message and exit
  --bbox MIN_LON MIN_LAT MAX_LON MAX_LAT
                        Bounding box to sample tiles from (min_lon min_lat max_lon max_lat)
  --zoom ZOOM           Zoom level of the tiles
  --sample_ratio SAMPLE_RATIO
                        Ratio of tiles to sample from the bounding box (between 0 and 1)
  --save_dir SAVE_DIR   Directory to save the tiles
  --url URL             Tile server base URL
```

## Example Workflow

```sh
# Step 1: Download pbf file (Here we only download the extract for Taiwan for demonstration)
wget -P ./work https://download.geofabrik.de/asia/taiwan-latest.osm.pbf
# Step 2: Slice the pbf file into smaller bounding boxes
bash slice_osm.sh \
    --planet-path ./work/taiwan-latest.osm.pbf \
    --slice-dir ./work/slices \
    --west 121.5 --east 121.6 \
    --south 25.0 --north 25.1 \
    --lon-step 0.05 --lat-step 0.05
# Step 3: Import the slices into PostgreSQL databases
bash import_osm.sh \
    --slice-dir ./work/slices \
    --volume-dir ./work/volumes \
    --import-split 0 \
    --total-splits 1
# Step 4: Run the tile server and scrape tiles
bash run.sh \
    --volume-dir ./work/volumes \
    --save-dir ./work/tiles \
    --zoom-level 15 \
    --sample-ratio 0.1 \
    --run-split 0 \
    --total-splits 1
```
