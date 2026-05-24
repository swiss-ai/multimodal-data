"""Download a grid of (layer x scale x resolution) samples for a few tiles for visual inspection."""

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode

import requests

WMS = "https://wms.geo.admin.ch/"
OUT = Path("/tmp/toolbox/swisstopo_maps/data/explore")
CSV_FILE = "/tmp/toolbox/swisstopo_maps/data/tiles_settlement_500m_overlap50m.csv"

LAYERS = [
    "ch.swisstopo.pixelkarte-farbe",
    "ch.swisstopo.pixelkarte-farbe-winter",
    "ch.swisstopo.swisstlm3d-karte-farbe",
    "ch.swisstopo-vd.amtliche-vermessung",
    "ch.kantone.cadastralwebmap-farbe",
    "ch.swisstopo.vec25-gebaeude",
    "ch.swisstopo.landeskarte-farbe-10",
    "ch.swisstopo.pixelkarte-farbe-pk100.noscale",
    "ch.swisstopo.pixelkarte-farbe-pk200.noscale",
    "ch.swisstopo.pixelkarte-farbe-pk500.noscale",
    "ch.swisstopo.pixelkarte-farbe-pk1000.noscale",
]

# scale multipliers around the 500m base box
SCALES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # 125m,250m,500m,1km,2km,4km on the ground

RESOLUTIONS = [(512, 512), (768, 768), (1024, 1024)]
LANGS = ["en", "de", "fr", "it"]


def expand_bbox(bbox_str, scale):
    a, b, c, d = [float(x) for x in bbox_str.split(",")]
    cx, cy = (a + c) / 2, (b + d) / 2
    half = (c - a) / 2 * scale
    return f"{cx - half},{cy - half},{cx + half},{cy + half}"


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
            return f"OK {path.name}"
        return f"ERR {r.status_code} {path.name}"
    except Exception as e:
        return f"FAIL {path.name}: {e}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Pick a few tiles at different positions in CSV
    rows = []
    with open(CSV_FILE) as f:
        r = csv.DictReader(f)
        rows = list(r)
    picks = [rows[1000], rows[20000], rows[60000]]  # diverse positions

    jobs = []

    # 1) For first tile: layer x scale grid at 768x768 EN
    t = picks[0]
    for layer in LAYERS:
        for scale in SCALES:
            bbox = expand_bbox(t["bbox"], scale)
            url = build_url(layer, bbox, 768, 768, "en")
            name = f"grid_{t['name']}_{layer}_s{scale}.png".replace(":", "_")
            jobs.append((url, OUT / name))

    # 2) Resolution comparison: pixelkarte-farbe at scale 1.0
    for w, h in RESOLUTIONS:
        bbox = expand_bbox(t["bbox"], 1.0)
        url = build_url("ch.swisstopo.pixelkarte-farbe", bbox, w, h, "en")
        jobs.append((url, OUT / f"res_{t['name']}_pixelkarte_{w}x{h}.png"))

    # 3) Language comparison: pixelkarte-farbe at 768
    for lang in LANGS:
        bbox = expand_bbox(t["bbox"], 1.0)
        url = build_url("ch.swisstopo.pixelkarte-farbe", bbox, 768, 768, lang)
        jobs.append((url, OUT / f"lang_{t['name']}_pixelkarte_{lang}.png"))

    # 4) Other tiles: just pixelkarte at scale 1 and 2
    for t in picks[1:]:
        for layer in [
            "ch.swisstopo.pixelkarte-farbe",
            "ch.kantone.cadastralwebmap-farbe",
            "ch.swisstopo.vec25-gebaeude",
        ]:
            for scale in [1.0, 2.0]:
                bbox = expand_bbox(t["bbox"], scale)
                url = build_url(layer, bbox, 768, 768, "en")
                jobs.append((url, OUT / f"div_{t['name']}_{layer}_s{scale}.png"))

    print(f"Downloading {len(jobs)} samples...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(download, jobs):
            if r:
                print(r)


if __name__ == "__main__":
    main()
