import requests
import csv


WMS_URL = "https://wms.geo.admin.ch/"
# load LV-95 coordinates
CSV_PATH = "downlaod_swisstopo_tiles_settlement_500m_overlap50m.csv"
# rs image path and map path
OUT_IMG = "./data/rs/"
OUT_MAP = "./data/map/"

def download_pair(bbox, width, height, crs="EPSG:2056", out_img="rs.png", out_map="map.png"):
    """
    bbox: "minX,minY,maxX,maxY" 
    width/height: image size
    crs: coordinate system
    out_img/out_map: output path
    """

    # 1) SWISSIMAGE
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

    r1 = requests.get(WMS_URL, params=params_img, timeout=60)
    r1.raise_for_status()
    with open(out_img, "wb") as f:
        f.write(r1.content)

    # 2) Swiss Map Raster 10
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

    r2 = requests.get(WMS_URL, params=params_map, timeout=60)
    r2.raise_for_status()
    with open(out_map, "wb") as f:
        f.write(r2.content)


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
    
    for i in range(len(jobs)):
        download_pair(bbox=jobs[i]['bbox'],width=1024, height=1024,crs="EPSG:2056", out_img=OUT_IMG+jobs[i]['name']+"_rs.png", out_map=OUT_MAP+jobs[i]['name']+"_map.png")
    
