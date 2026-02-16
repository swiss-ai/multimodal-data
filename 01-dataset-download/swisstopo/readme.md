# Swisstopo WMS Tile Downloader

Download paired tiles (aerial image + map) from the Swiss Federal Geoportal WMS service.

- **SWISSIMAGE** (aerial imagery)  
- **Swiss Map Raster 10** (topographic map)  
- WMS endpoint: https://wms.geo.admin.ch/  
- Coordinate system: EPSG:2056 (LV95)

---

## Requirements

- Python 3.8+
- requests

---

## Configurations

Edit paths inside the script if needed:

- CSV_PATH = "downlaod_swisstopo_tiles_settlement_500m_overlap50m.csv"
- OUT_IMG = "./data/rs/"
- OUT_MAP = "./data/map/"
