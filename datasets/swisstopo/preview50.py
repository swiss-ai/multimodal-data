"""Download 50 random PNGs that approximate the final distribution.

Uses the existing filtered_test.csv (200 prefiltered tiles) and the
weighted distributions from build_urls.py to produce one URL per tile,
then takes 50 at random.

Filenames embed layer + scale + frac so we can spot patterns.
"""

import csv
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import build_urls as bu
import requests

OUT = Path(os.environ.get("PREVIEW_OUT_DIR", "/tmp/toolbox/swisstopo_maps/data/preview50"))
TILES = os.environ.get("TILES_CSV", "/tmp/toolbox/swisstopo_maps/data/filtered_test.csv")
N = 50


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(3)

    with open(TILES, newline="") as f:
        tiles = list(csv.DictReader(f))
    rng.shuffle(tiles)

    # Build one config per tile for ALL filtered tiles, then take N random.
    # This gives a layer mix that matches the true population (cadastral
    # only fires on dense tiles, etc.).
    all_jobs = []
    for t in tiles:
        frac = float(t.get("building_frac", 0) or 0)
        layer = bu.weighted_choice(rng, bu.LAYER_WEIGHTS)
        if layer in bu.NONADAPTIVE_LAYERS and frac < bu.NONADAPTIVE_MIN_FRAC:
            layer = "ch.swisstopo.pixelkarte-farbe"
        scale = (
            bu.weighted_choice(rng, bu.SCALES_ADAPTIVE)
            if layer in bu.ADAPTIVE_LAYERS
            else bu.weighted_choice(rng, bu.SCALES_NONADAPTIVE)
        )
        res = bu.weighted_choice(rng, bu.RES_WEIGHTS)
        lang = bu.weighted_choice(rng, bu.LANG_WEIGHTS)
        bbox = bu.expand_bbox(t["bbox"], scale)
        url = bu.build_url(layer, bbox, res, res, lang)
        short = layer.split(".")[-1]
        name = f"{t['name']}_{short}_s{scale}_{res}_{lang}_f{frac:.3f}.png"
        all_jobs.append((url, OUT / name))

    rng.shuffle(all_jobs)
    jobs = all_jobs[:N]

    def fetch(args):
        url, path = args
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                path.write_bytes(r.content)
                return path.name
        except Exception as e:
            return f"FAIL {path.name}: {e}"

    print(f"Fetching {len(jobs)} previews to {OUT}")
    with ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(fetch, jobs):
            if r:
                print(r)


if __name__ == "__main__":
    main()
