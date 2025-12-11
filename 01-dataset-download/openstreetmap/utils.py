import numpy as np


def get_tile_coordinates(
    longitude: float, latitude: float, zoom: int
) -> tuple[int, int]:
    """Convert longitude and latitude to OpenStreetMap tile coordinates at a given zoom level.

    Args:
        longitude (float): the longitude (degree) of the north-west corner of the tile
        latitude (float): the latitude (degree) of the north-west corner of the tile
        zoom (int): the zoom level

    Returns:
        tuple[int, int]: the x and y tile coordinates
    """
    # x = np.radians(longitude)
    # y = np.arcsinh(np.tan(np.radians(latitude)))
    lat_rad = np.radians(latitude)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    eps = 1e-10
    tan_lat = sin_lat / (cos_lat + eps)
    sec_lat = 1 / (cos_lat + eps)
    n = 2**zoom
    x = n * ((longitude + 180) / 360)
    y = n * (1 - np.log(tan_lat + sec_lat + eps) / np.pi) / 2
    x = np.clip(x, 0, n - 1)
    y = np.clip(y, 0, n - 1)
    return int(np.floor(x)), int(np.floor(y))


def get_tile_lonlat(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Convert OpenStreetMap tile coordinates to longitude and latitude at a given zoom level.

    Args:
        x (int): x coordinate of the tile
        y (int): y coordinate of the tile
        zoom (int): zoom level of the tile
    Returns:
        tuple[float, float]: the longitude and latitude (degree) of the north-west corner of the tile
    """
    n = 2**zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
    lat_deg = np.degrees(lat_rad)
    return float(lon_deg), float(lat_deg)


def get_scale(latitude: float, zoom: int) -> float:
    """Get the scale (meters per pixel) at a given zoom level.

    Reference: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Resolution_and_Scale

    Args:
        latitude (float): the latitude (degree) of the location
        zoom (int): the zoom level

    Returns:
        float: the scale (meters per pixel)
    """
    zoom0_scale = 156543.03
    return zoom0_scale * np.cos(np.radians(latitude)) / (2**zoom)
