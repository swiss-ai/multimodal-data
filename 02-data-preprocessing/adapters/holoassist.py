import logging
import os
import shutil
import sys
import tarfile

import av
import cv2
import imagehash
from PIL import Image

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, ImageSample, SampleMetadata


def grid_laplacian_var(gray, grid_size):
    h, w = gray.shape
    rows, cols = grid_size

    dy, dx = h // rows, w // cols
    variances = []

    for i in range(rows):
        for j in range(cols):
            tile = gray[i * dy : (i + 1) * dy, j * dx : (j + 1) * dx]
            tile_var = cv2.Laplacian(tile, cv2.CV_64F).var()
            variances.append(tile_var)

    min_var = min(variances)
    avg_var = sum(variances) / len(variances)

    return min_var, avg_var


class HoloAssistAdapter(BaseDataset):
    def __init__(
        self,
        data_dir,
        min_laplacian_threshold,
        avg_laplacian_threshold,
        imagehash_tolerance,
        sample_rate=12,
    ):
        self.data_dir = data_dir
        self.min_laplacian_threshold = min_laplacian_threshold
        self.avg_laplacian_threshold = avg_laplacian_threshold
        self.imagehash_tolerance = imagehash_tolerance
        self.sample_rate = sample_rate

        self.member_list = []
        with tarfile.open(self.data_dir, "r") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".mp4"):
                    continue
                self.member_list.append(member)

    @property
    def id(self):
        return "holoassist"

    def stream(self, logger, skip=None, batch_size=1):
        if skip is not None:
            logger.warning("Skipping is not supported for HoloAssist.")
            # Supporting skip would require getting the number of accepted frames
            # in each video, which is non-trivial without processing all previous videos.
            # The better way is to delete the entire processed output from holoassist
            # and start fresh (since the dataset is not too large).
            assert skip == 0

        pending = []

        logger.info(f"Starting stream for {self.id} from {self.data_dir}")

        with tarfile.open(self.data_dir, "r") as tar:
            for mi, member in enumerate(self.member_list):
                logger.info(f"Processing video {mi + 1}/{len(self.member_list)}")
                f = tar.extractfile(member)
                container = av.open(f)
                container.streams.video[0].thread_type = "AUTO"

                last_sample_id = None
                kept_hashes = []
                for fi, frame in enumerate(container.decode(video=0)):  # type:ignore
                    if (
                        last_sample_id is not None
                        and last_sample_id + self.sample_rate >= fi
                    ):
                        continue

                    img = frame.to_ndarray(format="bgr24")
                    assert img.shape in [(504, 896, 3), (256, 454, 3)], img.shape

                    # blurriness check
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    min_var, avg_var = grid_laplacian_var(gray, grid_size=(4, 7))
                    if (
                        min_var < self.min_laplacian_threshold
                        or avg_var < self.avg_laplacian_threshold
                    ):
                        continue

                    # deduplication
                    is_duplicate = False
                    gray_pil = Image.fromarray(gray)
                    this_hash = imagehash.phash(gray_pil)
                    for kept_hash in reversed(kept_hashes):
                        if this_hash - kept_hash < self.imagehash_tolerance:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        continue

                    m = SampleMetadata(
                        dataset_id=self.id,
                        sample_id=fi,
                        data={
                            "dataset_id": self.id,
                            "video_name": member.name,
                            "timestamp": frame.time,
                        },
                    )

                    pil_img = frame.to_image().convert("RGB")
                    sample = ImageSample(meta=m, image=pil_img)

                    last_sample_id = fi
                    kept_hashes.append(this_hash)
                    pending.append(sample)
                    if len(pending) >= batch_size:
                        yield pending
                        pending = []

        if pending:
            yield pending

        logger.info("Finished streaming.")


if __name__ == "__main__":
    debug_path = "/iopsstor/scratch/cscs/tchu/debug/holoassist"
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    shutil.rmtree(debug_path, ignore_errors=True)
    os.makedirs(debug_path, exist_ok=True)

    a = HoloAssistAdapter(
        # data_dir="/capstor/store/cscs/swissai/infra01/vision-datasets/holoassist/video_pitch_shifted.tar",
        data_dir="/capstor/store/cscs/swissai/infra01/vision-datasets/holoassist/video_compress.tar",
        min_laplacian_threshold=10,
        avg_laplacian_threshold=1000,
        imagehash_tolerance=8,
    )

    total = 0
    for bi, batch in enumerate(a.stream(logger=logger, skip=0, batch_size=100)):
        total += len(batch)
        # for b in batch:
        #     b.image.save(f"{debug_path}/sample_{b.meta.sample_id:06d}.jpg")
        print(f"last sample_id in batch {bi}: {batch[-1].meta.sample_id}")
        # break
    print("Total samples:", total)
