import csv
import math
from urllib.parse import urlencode

# ----------------------------
# EPSG:4326 -> EPSG:3857
# ----------------------------
R = 6378137.0


def lonlat_to_webmercator(lon: float, lat: float):
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


# ----------------------------
# French cities
# ----------------------------
CITIES = {
    "Paris": (2.3522, 48.8566),
    "Marseille": (5.3698, 43.2965),
    "Lyon": (4.8357, 45.7640),
    "Toulouse": (1.4442, 43.6047),
    "Nice": (7.2619, 43.7102),
    "Nantes": (-1.5536, 47.2184),
    "Strasbourg": (7.7521, 48.5734),
    "Montpellier": (3.8767, 43.6110),
    "Bordeaux": (-0.5792, 44.8378),
    # "Lille": (3.0573, 50.6292),
}


# ----------------------------
# 500m*500m
# ----------------------------
def iter_tiles_3857(center_lon, center_lat, tile_size_m=500.0, radius_m=20000.0):
    cx, cy = lonlat_to_webmercator(center_lon, center_lat)

    minx = cx - radius_m
    maxx = cx + radius_m
    miny = cy - radius_m
    maxy = cy + radius_m

    cols = math.ceil((maxx - minx) / tile_size_m)
    rows = math.ceil((maxy - miny) / tile_size_m)

    for r in range(rows):
        for c in range(cols):
            x0 = minx + c * tile_size_m
            x1 = x0 + tile_size_m
            y0 = miny + r * tile_size_m
            y1 = y0 + tile_size_m
            yield (x0, y0, x1, y1, r, c)


def build_wms_url(base_url: str, params: dict) -> str:
    #
    return base_url + "?" + urlencode(params, safe=",:")  #


def main():
    # ===== params =====
    out_csv = "france_tiles_template.csv"

    tile_size_m = 1000.0
    radius_m = 20000.0
    width = 512
    height = 512
    transparent = "TRUE"
    img_format = "image/png"

    # ===== France WMS config =====
    wms_base = "https://data.geopf.fr/wms-r/wms"
    layer = "GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2"
    crs = "EPSG:3857"

    rows_out = []

    for city, (lon, lat) in CITIES.items():
        for minx, miny, maxx, maxy, r, c in iter_tiles_3857(lon, lat, tile_size_m=tile_size_m, radius_m=radius_m):
            bbox = f"{minx:.3f},{miny:.3f},{maxx:.3f},{maxy:.3f}"

            params = {
                "SERVICE": "WMS",
                "VERSION": "1.3.0",
                "REQUEST": "GetMap",
                "LAYERS": layer,
                "STYLES": "",
                "FORMAT": img_format,
                "CRS": crs,
                "BBOX": bbox,
                "WIDTH": str(width),
                "HEIGHT": str(height),
                "TRANSPARENT": transparent,
            }

            url = build_wms_url(wms_base, params)

            # url｜bbox
            rows_out.append(
                {
                    "url": url,
                    "bbox": bbox,
                }
            )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "bbox"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Saved: {out_csv}")
    print(f"Total rows: {len(rows_out)}")
    print("Example row:")
    if rows_out:
        print(rows_out[0]["url"], "|", rows_out[0]["bbox"])


if __name__ == "__main__":
    main()
