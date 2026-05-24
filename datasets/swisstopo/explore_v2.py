"""Phase-1 exploration v2.

Goals:
- For each non-pixelkarte layer, find the sweet-spot scale (visible labels,
  buildings still discernible).
- Verify which layers are scale-adaptive (rendering style changes at
  different scales) vs. just naive zoom.
- Produce per-layer subfolders for easy inspection.
"""

import csv
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode

import requests

WMS = os.environ.get("WMS_URL", "https://wms.geo.admin.ch/")
OUT = Path(os.environ.get("EXPLORE_OUT_DIR", "/tmp/toolbox/swisstopo_maps/data/explore_v2"))
FILTERED = os.environ.get("FILTERED_CSV", "/tmp/toolbox/swisstopo_maps/data/filtered_test.csv")

# Densest tiles from filtered_test.csv (building_frac):
#   tile_000176 = 0.121, tile_000197 = 0.055, tile_000200 = 0.028
TILES = ["tile_000176", "tile_000197", "tile_000200", "tile_000014"]

LAYERS = [
    "ch.swisstopo.pixelkarte-farbe",
    "ch.swisstopo.pixelkarte-farbe-winter",
    "ch.swisstopo.swisstlm3d-karte-farbe",
    "ch.swisstopo-vd.amtliche-vermessung",
    "ch.kantone.cadastralwebmap-farbe",
]

# fine-grained scales — small (zoomed in) is more useful per user
SCALES = [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]


def expand_bbox(bbox_str, scale):
    a, b, c, d = (float(x) for x in bbox_str.split(","))
    cx, cy = (a + c) / 2, (b + d) / 2
    half = (c - a) / 2 * scale
    return f"{cx - half:.2f},{cy - half:.2f},{cx + half:.2f},{cy + half:.2f}"


def build_url(layer, bbox, w, h, lang="en"):
    p = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": "image/png",
        "CRS": "EPSG:2056",
        "BBOX": bbox,
        "WIDTH": w,
        "HEIGHT": h,
        "TRANSPARENT": "false",
        "LANG": lang,
    }
    return WMS + "?" + urlencode(p)


def download(args):
    url, path = args
    if path.exists():
        return
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            path.write_bytes(r.content)
            return f"OK {path.relative_to(OUT)}"
        return f"ERR {r.status_code} {path.name}"
    except Exception as e:
        return f"FAIL {path.name}: {e}"


def main():
    with open(FILTERED, newline="") as f:
        rows = {r["name"]: r["bbox"] for r in csv.DictReader(f)}

    jobs = []
    for layer in LAYERS:
        layer_dir = OUT / layer.replace(":", "_")
        layer_dir.mkdir(parents=True, exist_ok=True)
        for tile in TILES:
            if tile not in rows:
                continue
            for s in SCALES:
                bbox = expand_bbox(rows[tile], s)
                url = build_url(layer, bbox, 768, 768)
                name = f"{tile}_s{s}.png"
                jobs.append((url, layer_dir / name))

    print(f"Downloading {len(jobs)} samples to {OUT}")
    with ThreadPoolExecutor(max_workers=24) as ex:
        for r in ex.map(download, jobs):
            if r:
                print(r)


if __name__ == "__main__":
    main()
