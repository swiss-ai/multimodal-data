"""Pre-filter tiles by querying ch.swisstopo.vec25-gebaeude.

For each tile, request a small (256x256) vec25 image, count the fraction
of non-white pixels (= buildings), and keep tiles with >= MIN_FRACTION.
A KEEP_EMPTY_RATIO fraction of tiles with no/few buildings is kept too,
so the final dataset can also caption empty/sparse areas.

Output: a CSV with the same schema as the input plus a `building_frac` column.
"""

import argparse
import csv
import io
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode

import requests
from PIL import Image

WMS = "https://wms.geo.admin.ch/"
LAYER = "ch.swisstopo.vec25-gebaeude"
PROBE_RES = 256
MIN_FRACTION = 0.003  # 0.3%
KEEP_EMPTY_RATIO = 0.0
TIMEOUT = 30


def probe_url(bbox):
    p = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": LAYER,
        "STYLES": "",
        "FORMAT": "image/png",
        "CRS": "EPSG:2056",
        "BBOX": bbox,
        "WIDTH": PROBE_RES,
        "HEIGHT": PROBE_RES,
        "TRANSPARENT": "false",
    }
    return WMS + "?" + urlencode(p)


def building_fraction(bbox):
    r = requests.get(probe_url(bbox), timeout=TIMEOUT)
    if r.status_code != 200:
        return -1.0
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    # downsample to 64x64 for speed; count pixels not near-white
    small = img.resize((64, 64))
    data = small.tobytes()
    non_white = total = 0
    for i in range(0, len(data), 3):
        total += 1
        if data[i] < 250 or data[i + 1] < 250 or data[i + 2] < 250:
            non_white += 1
    return non_white / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="data/tiles_settlement_500m_overlap50m.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    with open(args.tiles, newline="") as f:
        tiles = list(csv.DictReader(f))
    if args.limit:
        tiles = tiles[: args.limit]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    kept = dropped = errored = empty_kept = 0
    with open(args.out, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["name", "bbox", "building_frac"])
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            futs = {ex.submit(building_fraction, t["bbox"]): t for t in tiles}
            for i, fut in enumerate(as_completed(futs)):
                t = futs[fut]
                try:
                    frac = fut.result()
                except Exception:
                    frac = -1.0
                if frac < 0:
                    errored += 1
                    continue
                if frac >= MIN_FRACTION:
                    kept += 1
                    w.writerow([t["name"], t["bbox"], f"{frac:.5f}"])
                elif rng.random() < KEEP_EMPTY_RATIO:
                    empty_kept += 1
                    kept += 1
                    w.writerow([t["name"], t["bbox"], f"{frac:.5f}"])
                else:
                    dropped += 1
                if (i + 1) % 1000 == 0:
                    print(
                        f"{i + 1}/{len(tiles)} kept={kept} dropped={dropped} empty_kept={empty_kept} err={errored}",
                        flush=True,
                    )
    print(f"DONE total={len(tiles)} kept={kept} dropped={dropped} empty_kept={empty_kept} err={errored}")


if __name__ == "__main__":
    main()
