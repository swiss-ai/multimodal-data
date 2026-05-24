import io
import os
from collections import deque

import cv2
import h5py
import numpy as np
import webdataset as wds
from PIL import Image
from tqdm import tqdm

LAPLACIAN_THRESHOLD = 3.5
SAMPLING_RATE = 18
HASH_SQRT_SIZE = 8
SHARD_SIZE = 1000

ORIGINAL_SIZE = (3840, 2160)
RESIZED_SIZE = (768, 432)


def is_blurry(image_bytes, threshold=LAPLACIAN_THRESHOLD):
    img_np = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var <= threshold


def pil_to_jpeg_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=100)
    return buf.getvalue()


def avg_hash(img):
    img = img.resize((HASH_SQRT_SIZE, HASH_SQRT_SIZE), Image.LANCZOS).convert("L")
    pixels = np.array(img).flatten()
    avg = pixels.mean()
    hash_str = "".join(["1" if pixel > avg else "0" for pixel in pixels])
    return hash_str


output_dir = os.environ.get("OUTPUT_DIR", "output_wds2")
if os.path.exists(output_dir):
    raise FileExistsError(f"Output directory '{output_dir}' already exists")
os.makedirs(output_dir, exist_ok=True)

with (
    h5py.File(os.environ.get("HDF5_PATH", "RGB_dataset.hdf5"), "r") as f,
    wds.ShardWriter(os.path.join(output_dir, "shard-%06d.tar"), maxcount=SHARD_SIZE) as sink,
):
    color_grp = f["color"]
    for scene_name in tqdm(color_grp.keys(), desc="Scenes", leave=True):
        scene_grp = color_grp[scene_name]
        for video_name in tqdm(scene_grp.keys(), desc="Videos", leave=True):
            video_grp = scene_grp[video_name]

            counter = SAMPLING_RATE
            hashes = deque(maxlen=3)
            cache = []

            for image_name in tqdm(video_grp.keys(), desc="Images", leave=False):
                counter += 1
                if counter <= SAMPLING_RATE:
                    continue

                dataset = video_grp[image_name]
                binary_data_np = dataset[()]
                raw_bytes = binary_data_np.item()

                # blurry check
                if is_blurry(raw_bytes):
                    continue

                # resize
                img_pil = Image.open(io.BytesIO(raw_bytes))
                assert img_pil.size == ORIGINAL_SIZE
                img_pil = img_pil.resize(RESIZED_SIZE, Image.LANCZOS)

                # hash check
                img_hash = avg_hash(img_pil)
                if img_hash in hashes:
                    continue
                hashes.append(img_hash)
                counter = 0

                cache.append(img_pil)
                if len(cache) == 3:
                    img_num = int(image_name)
                    scene_name = scene_name.replace(".", "_")
                    video_name = video_name.replace(".", "_")
                    key = f"{scene_name}_{video_name}_{img_num:06}"
                    sink.write(
                        {
                            "__key__": key,
                            "img0.jpg": pil_to_jpeg_bytes(cache[0]),
                            "img1.jpg": pil_to_jpeg_bytes(cache[1]),
                            "img2.jpg": pil_to_jpeg_bytes(cache[2]),
                        }
                    )
                    cache.clear()
                    hashes.clear()
                    cache.append(img_pil)
                    hashes.append(img_hash)
