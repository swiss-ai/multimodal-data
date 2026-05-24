"""Generate WMS URLs for img2dataset.

Reads a tile CSV (with `name`, `bbox`, `building_frac`) and writes
one URL per row (sample_id, tile_id, bbox, layer, img_w, img_h, lang,
scale, url, building_frac).

Per tile we sample N (default 3) configurations. Layer choice is
weighted; the **scale distribution depends on the layer**:

- Adaptive layers (`pixelkarte-farbe`, `*-winter`, `swisstlm3d-karte-farbe`)
  use the full 0.5x–4x range. The server changes cartographic style with
  scale, so each scale gives genuinely different visual content.
- Non-adaptive layers (`cadastralwebmap-farbe`, `amtliche-vermessung`)
  have fixed-size pixel typography that becomes unreadable when zoomed
  out. They are only emitted at very small scales (0.2x–0.3x).

Cadastral is also gated on building density: tiles with
`building_frac < CADASTRAL_MIN_FRAC` resample as pixelkarte-farbe instead.
"""

import argparse
import csv
import os
import random
from pathlib import Path
from urllib.parse import urlencode

WMS = os.environ.get("WMS_URL", "https://wms.geo.admin.ch/")

LAYER_WEIGHTS = {
    "ch.swisstopo.pixelkarte-farbe": 0.675,
    "ch.kantone.cadastralwebmap-farbe": 0.28,
    "ch.swisstopo-vd.amtliche-vermessung": 0.02,
    "ch.swisstopo.pixelkarte-farbe-winter": 0.02,
    "ch.swisstopo.swisstlm3d-karte-farbe": 0.005,
}

ADAPTIVE_LAYERS = {
    "ch.swisstopo.pixelkarte-farbe",
    "ch.swisstopo.pixelkarte-farbe-winter",
    "ch.swisstopo.swisstlm3d-karte-farbe",
}

# Multiplier of the 500 m base box: 0.75x=375m, 1.0x=500m, 2.0x=1km, 4.0x=2km
# Pixelkarte / winter / 3d are unreadable below 0.75x.
SCALES_ADAPTIVE = {0.75: 0.15, 1.0: 0.50, 2.0: 0.25, 4.0: 0.10}
# Non-adaptive layers: 0.75x mostly, 0.5x for variety. 375m and 250m bboxes.
SCALES_NONADAPTIVE = {0.75: 0.85, 0.5: 0.15}

RES_WEIGHTS = {1024: 0.70, 768: 0.25, 512: 0.05}

# Cadastral typography is too small at <1024 — always render at 1024.
FORCE_1024_LAYERS = {"ch.kantone.cadastralwebmap-farbe"}

# Bbox center jitter (in units of base half-bbox = 250 m). Per-tile
# offsets come from a Halton(2,3) quasi-random sequence so the N samples
# within a tile spread across the jitter square instead of clustering by
# chance. Each tile picks a random Halton start index so tiles don't
# share the same pattern.
JITTER_RANGE = 0.5  # ±125 m around the tile centre

# Fallback language weights used when samples-per-tile != 8. At 8 we
# enforce a fixed 5/1/1/1 quota instead.
LANG_WEIGHTS = {"en": 0.85, "de": 0.05, "fr": 0.05, "it": 0.05}
LANG_QUOTA_8 = ["en"] * 5 + ["de", "fr", "it"]

# Non-adaptive layers (cadastralwebmap, amtliche-vermessung) require a
# minimum building density to be worthwhile. Below this, the tile produces
# mostly empty parcel diagrams — fall back to pixelkarte instead.
NONADAPTIVE_MIN_FRAC = 0.05

NONADAPTIVE_LAYERS = {
    "ch.kantone.cadastralwebmap-farbe",
    "ch.swisstopo-vd.amtliche-vermessung",
}


def weighted_choice(rng, dist):
    keys = list(dist.keys())
    return rng.choices(keys, weights=[dist[k] for k in keys], k=1)[0]


def expand_bbox(bbox_str, scale, dx=0.0, dy=0.0):
    a, b, c, d = (float(x) for x in bbox_str.split(","))
    base_half = (c - a) / 2
    cx = (a + c) / 2 + dx * base_half
    cy = (b + d) / 2 + dy * base_half
    half = base_half * scale
    return f"{cx - half:.2f},{cy - half:.2f},{cx + half:.2f},{cy + half:.2f}"


def build_url(layer, bbox, w, h, lang):
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


def pick_scale(rng, layer):
    dist = SCALES_ADAPTIVE if layer in ADAPTIVE_LAYERS else SCALES_NONADAPTIVE
    return weighted_choice(rng, dist)


def halton(i, base):
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def jitter_offsets(rng, n):
    """Return n (dx, dy) pairs in [-JITTER_RANGE, +JITTER_RANGE]^2,
    well-spread via a Halton(2,3) sequence with a random start index."""
    start = rng.randint(1, 10_000_000)
    out = []
    for k in range(n):
        u = halton(start + k, 2)
        v = halton(start + k, 3)
        out.append(((u - 0.5) * 2 * JITTER_RANGE, (v - 0.5) * 2 * JITTER_RANGE))
    rng.shuffle(out)  # decorrelate jitter index from sample index k
    return out


def lang_assignments(rng, n):
    if n == 8:
        langs = LANG_QUOTA_8.copy()
        rng.shuffle(langs)
        return langs
    return [weighted_choice(rng, LANG_WEIGHTS) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, help="CSV with columns name,bbox,building_frac")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples-per-tile", type=int, default=3)
    ap.add_argument("--limit-tiles", type=int, default=0)
    ap.add_argument("--shuffle-tiles", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    with open(args.tiles, newline="") as f:
        tiles = list(csv.DictReader(f))

    if args.shuffle_tiles:
        rng.shuffle(tiles)
    if args.limit_tiles > 0:
        tiles = tiles[: args.limit_tiles]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    nonadaptive_fallbacks = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sample_id",
                "tile_id",
                "bbox",
                "layer",
                "img_w",
                "img_h",
                "lang",
                "scale",
                "url",
                "building_frac",
            ]
        )
        for t in tiles:
            frac = float(t.get("building_frac", 0) or 0)
            offsets = jitter_offsets(rng, args.samples_per_tile)
            langs = lang_assignments(rng, args.samples_per_tile)
            for k in range(args.samples_per_tile):
                layer = weighted_choice(rng, LAYER_WEIGHTS)
                # Gate non-adaptive layers on building density
                if layer in NONADAPTIVE_LAYERS and frac < NONADAPTIVE_MIN_FRAC:
                    layer = "ch.swisstopo.pixelkarte-farbe"
                    nonadaptive_fallbacks += 1
                scale = pick_scale(rng, layer)
                if layer in FORCE_1024_LAYERS:
                    res = 1024
                else:
                    res = weighted_choice(rng, RES_WEIGHTS)
                lang = langs[k]
                dx, dy = offsets[k]
                bbox = expand_bbox(t["bbox"], scale, dx, dy)
                url = build_url(layer, bbox, res, res, lang)
                sid = f"{t['name']}_{k:02d}"
                w.writerow(
                    [
                        sid,
                        t["name"],
                        bbox,
                        layer,
                        res,
                        res,
                        lang,
                        scale,
                        url,
                        f"{frac:.5f}",
                    ]
                )
                n += 1
    print(f"wrote {n} urls to {args.out} (nonadaptive->pixelkarte fallbacks: {nonadaptive_fallbacks})")


if __name__ == "__main__":
    main()
