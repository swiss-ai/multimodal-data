import glob
import os

import numpy as np
import PIL.Image as Image
import torch
import utils


class OSMTileDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str):
        """Dataset for OpenStreetMap raster tiles stored in a local directory.

        Args:
            data_dir (str): Directory containing the raster tiles
        """
        self.data_dir = data_dir
        self.tile_list = self.load_tile_list()

    def load_tile_list(self) -> list[tuple[int, int, int]]:
        # Load the list of tiles from the data directory
        tile_files = glob.glob(os.path.join(self.data_dir, "*_*_*.png"))
        tile_list = []
        for tile_file in tile_files:
            basename = os.path.basename(tile_file)
            zoom, x, y = map(int, basename[:-4].split("_"))
            tile_list.append((zoom, x, y))
        return tile_list

    def __len__(self) -> int:
        return len(self.tile_list)

    def __getitem__(self, idx: int) -> tuple[Image.Image, str]:
        zoom, x, y = self.tile_list[idx]
        tile_path = os.path.join(self.data_dir, f"{zoom}_{x}_{y}.png")
        tile_image = Image.open(tile_path).convert("RGB")
        tile_image = tile_image
        description = self._get_description(zoom, x, y)
        return tile_image, description

    @staticmethod
    def _get_description(zoom: int, x: int, y: int) -> str:
        lon, lat = utils.get_tile_lonlat(x, y, zoom)
        scale = utils.get_scale(lat, zoom)
        return f"Tile at longitude {lon}, latitude {lat}, scale {scale:.2f} m/px"


if __name__ == "__main__":
    dataset = OSMTileDataset(data_dir="./samples/")
    for i in range(len(dataset)):
        tile_image, description = dataset[i]
        tile_image = np.array(tile_image)
        print(f"Tile {i}: {description}, Image shape: {tile_image.shape}")
