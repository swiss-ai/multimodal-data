import json
import logging
import os
import shutil
import sys
import time

import av
import cv2
import imagehash
import numpy as np
from PIL import Image

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, ImageSample, SampleMetadata


def get_circle_mask(size, radius_ratio=0.8):
    center = (size // 2, size // 2)
    radius = int(size * radius_ratio / 2)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    bool_mask = mask == 255
    return bool_mask


class EgoExo4DAdapter(BaseDataset):
    def __init__(
        self,
        data_dir,
        aria_map_x_path,
        aria_map_y_path,
        size,
        laplacian_threshold,
        imagehash_tolerance,
    ):
        self.data_dir = data_dir
        self.size = size
        self.laplacian_threshold = laplacian_threshold
        self.imagehash_tolerance = imagehash_tolerance

        takes_path = os.path.join(data_dir, "takes.json")
        with open(takes_path, "r") as f:
            takes = json.load(f)

        self.take_paths = []
        for take in takes:
            assert "take_name" in take
            assert "root_dir" in take
            assert "frame_aligned_videos" in take
            video_keys = take["frame_aligned_videos"].keys()
            aria_keys = [k for k in video_keys if "aria" in k.lower()]
            assert len(aria_keys) == 1, f"{take}"
            aria_key = aria_keys[0]
            assert "rgb" in take["frame_aligned_videos"][aria_key]
            assert "relative_path" in take["frame_aligned_videos"][aria_key]["rgb"]

            self.take_paths.append(
                os.path.join(
                    data_dir,
                    take["root_dir"],
                    take["frame_aligned_videos"][aria_key]["rgb"]["relative_path"],
                )
            )

        self.map_x = np.load(aria_map_x_path)
        self.map_y = np.load(aria_map_y_path)
        self.mask_bool = get_circle_mask(size)

    @property
    def id(self):
        return "egoexo4d"

    def stream(self, logger, skip=None, batch_size=1):
        _ = skip  # not implemented
        pending = []

        total_videos = len(self.take_paths)
        for vi, video_path in enumerate(self.take_paths):
            with av.open(video_path) as container:
                logger.info(f"Processing video: {vi + 1}/{total_videos}")

                stream = container.streams.video[0]
                stream.thread_type = "AUTO"

                kept_hashes = []

                for i, frame in enumerate(container.decode(video=0)):  # type:ignore
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.remap(
                        img, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR
                    )
                    img = cv2.resize(
                        img,
                        (self.size, self.size),
                        interpolation=cv2.INTER_LANCZOS4,
                    )

                    # blurriness check
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    score = laplacian[self.mask_bool].var()
                    if score < self.laplacian_threshold:
                        continue

                    # dedup
                    gray_pil = Image.fromarray(gray)
                    this_hash = imagehash.dhash(gray_pil)
                    is_duplicate = False
                    for seen_hash in reversed(kept_hashes):
                        if this_hash - seen_hash < self.imagehash_tolerance:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        continue

                    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    meta = SampleMetadata(
                        dataset_id=self.id,
                        sample_id=i,
                        data={
                            "video_path": video_path,
                            "frame_index": i,
                        },
                    )
                    sample = ImageSample(meta=meta, image=img_pil)

                    kept_hashes.append(this_hash)
                    pending.append(sample)
                    if len(pending) >= batch_size:
                        yield pending
                        pending = []

        if pending:
            yield pending

        logger.info(f"Finished streaming dataset {self.id}.")


if __name__ == "__main__":
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    shutil.rmtree("/iopsstor/scratch/cscs/tchu/debug/egoexo4d", ignore_errors=True)
    os.makedirs("/iopsstor/scratch/cscs/tchu/debug/egoexo4d", exist_ok=True)

    a = EgoExo4DAdapter(
        data_dir="/capstor/store/cscs/swissai/infra01/vision-datasets/egoexo4D",
        aria_map_x_path="/iopsstor/scratch/cscs/tchu/ego/aria_maps/egoexo4d/map_x.npy",
        aria_map_y_path="/iopsstor/scratch/cscs/tchu/ego/aria_maps/egoexo4d/map_y.npy",
        size=512,
        laplacian_threshold=400,
        imagehash_tolerance=5,
    )

    total = 0
    t0 = time.time()
    for batch in a.stream(logger=logger, batch_size=100):
        total += len(batch)
        for b in batch:
            print("obtained sample:", b.meta.sample_id, b.image.size)
            b.image.save(
                f"/iopsstor/scratch/cscs/tchu/debug/egoexo4d/sample_{b.meta.sample_id:06d}.jpg"
            )
        t1 = time.time()
        print("Total so far:", total, "Speed:", total / (t1 - t0), "samples/sec")
        break
    print("Total samples:", total)
