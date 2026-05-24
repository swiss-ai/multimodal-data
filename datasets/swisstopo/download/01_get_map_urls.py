import csv
from urllib.parse import urlencode

wms_url = "https://wms.geo.admin.ch/"
input_file = "./data/swisstopo_tiles_settlement_500m_overlap50m.csv"
output_file = "./data/map_urls.csv"

layers = [
    "ch.swisstopo.pixelkarte-farbe",
    "ch.swisstopo.pixelkarte-farbe-winter",
    "ch.swisstopo-vd.amtliche-vermessung",
    "ch.swisstopo-vd.geometa-standav",
    "ch.swisstopo.swisstlm3d-karte-farbe",
    "ch.are.bauzonen",  # to check if building zone
    "ch.swisstopo.vec25-gebaeude",  # to check if building zone
    #
    "ch.swisstopo.landeskarte-farbe-10",
    "ch.swisstopo.pixelkarte-farbe-pk100.noscale",
    "ch.swisstopo.pixelkarte-farbe-pk200.noscale",
    "ch.swisstopo.pixelkarte-farbe-pk500.noscale",
    "ch.swisstopo.pixelkarte-farbe-pk1000.noscale",
]


def build_url(sample_id, bbox, width, height, crs):
    """
    bbox: "minX,minY,maxX,maxY"
    width/height: image size
    crs: coordinate system
    out_img/out_map: output path
    """
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
    url = wms_url + "?" + urlencode(params_map)
    return {
        "id": sample_id,
        "bbox": bbox,
        "url": url,
    }


def read_jobs_from_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


if __name__ == "__main__":
    rows = read_jobs_from_csv(input_file)

    urls = []
    for row_id in range(len(rows)):
        urls.append(
            build_url(
                sample_id=rows[row_id]["name"],
                bbox=rows[row_id]["bbox"],
                width=768,
                height=768,
                crs="EPSG:2056",
            )
        )

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "bbox", "url"])
        writer.writeheader()
        writer.writerows(urls)
