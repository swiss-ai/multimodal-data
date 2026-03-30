import random

from lib.scrape import fetch_raster_tile
from lib.utils import get_tile_coordinates
from tqdm.auto import tqdm


def sample_from_bounding_box(
    bbox: tuple[float, float, float, float], zoom: int, sample_ratio: float
) -> list[tuple[int, int]]:
    """Samples tile coordinates from a bounding box.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box in the format (min_lon, min_lat, max_lon, max_lat)
        zoom (int): Zoom level
        sample_ratio (float): Ratio of tiles to sample from the total available tiles
    Returns:
        list[tuple[int, int]]: List of sampled tile coordinates (x, y)
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    min_x, min_y = get_tile_coordinates(min_lon, max_lat, zoom)
    max_x, max_y = get_tile_coordinates(max_lon, min_lat, zoom)

    max_n_tiles = (max_x - min_x + 1) * (max_y - min_y + 1)
    n_samples = int(max_n_tiles * sample_ratio)
    print(f"Sampling {n_samples}/{max_n_tiles} ({sample_ratio:.2%}) tiles.")
    print(f"Tile X range: {min_x} to {max_x}, Tile Y range: {min_y} to {max_y}")

    all_tiles = [
        (x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)
    ]

    random.seed(0)
    random.shuffle(all_tiles)
    sampled_tiles = all_tiles[:n_samples]
    sampled_tiles.sort()
    return sampled_tiles


def main(
    bounding_box: tuple[float, float, float, float] | None,
    zoom: int,
    sample_ratio: float,
    save_dir: str,
    server_url: str | None = None,
):
    if bounding_box is None:
        bounding_box = (120.0, 21.0 + 53 / 60, 122.0, 25.0 + 18 / 60)

    sampled_tiles = sample_from_bounding_box(
        bbox=bounding_box, zoom=zoom, sample_ratio=sample_ratio
    )
    for x, y in tqdm(sampled_tiles, leave=False):
        fetch_raster_tile(x, y, zoom, save_dir, server_url)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch raster tiles from a tile server."
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box to sample tiles from (min_lon min_lat max_lon max_lat)",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        required=True,
        help="Zoom level of the tiles",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="Ratio of tiles to sample from the bounding box (between 0 and 1)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the tiles",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Tile server base URL",
    )
    args = parser.parse_args()
    main(
        bounding_box=tuple(args.bbox) if args.bbox else None,
        zoom=args.zoom,
        sample_ratio=args.sample_ratio,
        save_dir=args.save_dir,
        server_url=args.url,
    )
