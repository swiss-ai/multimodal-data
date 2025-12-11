import os
import random
import time
from urllib.parse import urljoin

import requests

OPEN_TOPO_SERVERS = [
    "https://a.tile.opentopomap.org/",
    "https://b.tile.opentopomap.org/",
    "https://c.tile.opentopomap.org/",
]


def get_tile_url(x: int, y: int, zoom: int, server_url: str | None = None) -> str:
    """Constructs the URL for a raster tile.

    Format: {server}/{zoom}/{x}/{y}.png

    Args:
        x (int): x coordinate of the tile
        y (int): y coordinate of the tile
        zoom (int): zoom level of the tile
        server (str | None, optional): Tile server base URL. If None, a random server from opentopomap is chosen. Defaults to None.

    Returns:
        str: URL of the raster tile
    """
    tile_path = f"{zoom}/{x}/{y}.png"
    if server_url is None:
        server_url = random.choice(OPEN_TOPO_SERVERS)
    assert isinstance(server_url, str)
    return urljoin(server_url, tile_path)


def get_tile_local_path(x: int, y: int, zoom: int, local_dir: str) -> str:
    """Constructs the local file path for a raster tile.
    Args:
        x (int): x coordinate of the tile
        y (int): y coordinate of the tile
        zoom (int): zoom level of the tile
        local_dir (str): directory to save the tile
    Returns:
        str: Local file path of the raster tile
    """
    return os.path.join(local_dir, f"{zoom}_{x}_{y}.png")


def fetch_raster_tile(
    x: int, y: int, zoom: int, save_dir: str, server_url: str | None = None
) -> str:
    """Fetches a raster tile from a tile server and saves it locally.

    Tiles are cached locally to avoid redundant network requests.
    This function also includes a delay to avoid overwhelming the tile server.
    Each tile is a 256x256 PNG image.

    Args:
        x (int): x coordinate of the tile
        y (int): y coordinate of the tile
        zoom (int): zoom level of the tile
        save_dir (str): directory to save the tile
        server_url (str | None, optional): Tile server base URL. If None, a random server from opentopomap is chosen. Defaults to None.

    Returns:
        str: Path to the saved tile image file
    """
    url = get_tile_url(x, y, zoom, server_url)
    tile_save_path = get_tile_local_path(x, y, zoom, save_dir)

    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(tile_save_path):
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch tile {x}, {y}, {zoom} from {url}")
            print(f"Status code: {response.status_code}")
            return ""
        with open(tile_save_path, "wb") as f:
            f.write(response.content)

    return tile_save_path
