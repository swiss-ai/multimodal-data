"""
Generate WMS GetMap URLs for SWISSIMAGE tiles.

The input CSV must have a `bbox` column with values in LV-95 (EPSG:2056)
format: "minX,minY,maxX,maxY".

Usage:
    python generate_urls.py \
        --input-csv tiles.csv \
        --output-csv swissimage_urls.csv \
        --width 1024 \
        --height 1024
"""

import argparse
import csv
import uuid
from urllib.parse import urlencode

WMS_URL = "https://wms.geo.admin.ch/"
LAYER_SATELLITE = "ch.swisstopo.swissimage"


def build_wms_url(bbox: str, width: int, height: int, crs: str = "EPSG:2056") -> str:
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": LAYER_SATELLITE,
        "STYLES": "",
        "FORMAT": "image/png",
        "CRS": crs,
        "BBOX": bbox,
        "WIDTH": width,
        "HEIGHT": height,
        "TRANSPARENT": "false",
    }
    return WMS_URL + "?" + urlencode(params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, help="Input tile grid CSV")
    parser.add_argument("--output-csv", required=True, help="Output URL CSV")
    parser.add_argument("--width", type=int, default=1024, help="Image width in px")
    parser.add_argument("--height", type=int, default=1024, help="Image height in px")
    parser.add_argument("--crs", default="EPSG:2056", help="Coordinate reference system")
    args = parser.parse_args()

    rows = []
    with open(args.input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tile_id = str(uuid.uuid4())
            url = build_wms_url(row["bbox"], args.width, args.height, args.crs)
            rows.append({"id": tile_id, "url": url, "bbox": row["bbox"]})

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "url", "bbox"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows):,} URLs → {args.output_csv}")


if __name__ == "__main__":
    main()
