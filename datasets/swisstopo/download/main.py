import csv
import os
import uuid
from urllib.parse import urlencode

WMS_URL = os.environ.get("WMS_URL", "https://wms.geo.admin.ch/")
CSV_PATH = os.environ.get("TILES_CSV", "data/download_swisstopo_tiles_settlement_500m_overlap50m.csv")
OUT_CSV = os.environ.get("OUT_CSV", "./data/swisstopo_urls.csv")


def build_url_pair(bbox, width, height, crs="EPSG:2056"):
    """
    bbox: "minX,minY,maxX,maxY"
    width/height: image size
    crs: coordinate system
    """
    pair_id = str(uuid.uuid4())
    urls = []

    params_img = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": "ch.swisstopo.swissimage",
        "STYLES": "",
        "FORMAT": "image/png",
        "CRS": crs,
        "BBOX": bbox,
        "WIDTH": width,
        "HEIGHT": height,
        "TRANSPARENT": "false",
    }
    url_img = WMS_URL + "?" + urlencode(params_img)
    urls.append(
        {
            "id": str(uuid.uuid4()),
            "url": url_img,
            "pair_id": pair_id,
            "image_type": "satellite",
            "bbox": bbox,
        }
    )

    params_map = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": "ch.swisstopo.landeskarte-farbe-10",
        "STYLES": "",
        "FORMAT": "image/png",
        "CRS": crs,
        "BBOX": bbox,
        "WIDTH": width,
        "HEIGHT": height,
        "TRANSPARENT": "true",
    }
    url_map = WMS_URL + "?" + urlencode(params_map)
    urls.append(
        {
            "id": str(uuid.uuid4()),
            "url": url_map,
            "pair_id": pair_id,
            "image_type": "map",
            "bbox": bbox,
        }
    )

    return urls


def read_jobs_from_csv(csv_path):
    jobs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jobs.append(row)
    return jobs


if __name__ == "__main__":
    jobs = read_jobs_from_csv(CSV_PATH)
    jobs = list(jobs)

    all_urls = []
    for i in range(len(jobs)):
        urls = build_url_pair(
            bbox=jobs[i]["bbox"],
            width=1024,
            height=1024,
            crs="EPSG:2056",
        )
        all_urls.extend(urls)

    fieldnames = ["id", "url", "pair_id", "image_type", "bbox"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_urls)
