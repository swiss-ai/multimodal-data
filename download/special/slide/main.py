# import os
# import zipfile
# import PIL.Image
#
# dir_path = "/path/to/data/medical/raw/slide"
#
# for archive in archives:
#     jpg_files = [f for f in archive.namelist() if f.endswith(".jpg")]
#     for jpg_file in jpg_files:
#         with archive.open(jpg_file) as file:
#             image = PIL.Image.open(file)
#             image = image.resize((768, 584))


import io
import os
import zipfile

import PIL.Image
from torch.utils.data import DataLoader, Dataset


class SLIDEAdapter(Dataset):
    def __init__(self, dir_path, img_size=(768, 584)):
        self.dir_path = dir_path
        self.img_size = img_size

        zip_files = ["test.zip", "train.zip", "val.zip"]
        zip_paths = [os.path.join(dir_path, zip_file) for zip_file in zip_files]

        self.data_map = []
        for zip_path in zip_paths:
            with zipfile.ZipFile(zip_path, "r") as archive:
                self.data_map.extend([(zip_path, f) for f in archive.namelist() if f.endswith(".jpg")])

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        zip_path, file_name = self.data_map[idx]
        with zipfile.ZipFile(zip_path, "r") as archive:
            with archive.open(file_name) as file:
                img_data = file.read()
        image = PIL.Image.open(io.BytesIO(img_data))
        image = image.resize(self.img_size)
        return image

    @staticmethod
    def collate_fn(batch):
        return batch


# --- Usage ---

dataset = SLIDEAdapter(dir_path="/path/to/data/medical/raw/slide")

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    collate_fn=SLIDEAdapter.collate_fn,
)

# Test it
if __name__ == "__main__":
    total = 0
    for batch in loader:
        total += len(batch)
        print(f"Loaded a batch of {len(batch)} images")
    print(f"Total images loaded so far: {total}")
