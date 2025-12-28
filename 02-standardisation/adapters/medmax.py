import glob
import os
import subprocess
import tarfile

from PIL import Image

from pipeline import BaseDataset, ImageSample, SampleMetadata


class MedMaxRawImageAdapter(BaseDataset):
    def __init__(self, data_dir):
        search_pattern = os.path.join(data_dir, "images.tar.gz.*")
        self.chunk_files = sorted(glob.glob(search_pattern))

        if not self.chunk_files:
            raise FileNotFoundError(f"No tar chunks found in {data_dir}.'")

    @property
    def id(self):
        return "medmax_raw_images"

    def stream(self, logger, skip: int | None = None):
        for file in self.chunk_files:
            logger.info(f"Found chunk file: {file}")

        process = subprocess.Popen(
            ["cat"] + self.chunk_files,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if not skip:
            skip = 0

        logger.info(f"Starting to stream images, skipping first {skip} samples.")

        try:
            with tarfile.open(fileobj=process.stdout, mode="r|gz") as tar:
                counter = 0

                for member in tar:
                    if not member.isfile():
                        continue
                    if not member.name.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue
                    if counter < skip:
                        counter += 1
                        continue

                    try:
                        # load image from tar member to PIL Image
                        f_obj = tar.extractfile(member)
                        if not f_obj:
                            continue
                        pil_image = Image.open(f_obj).convert("RGB")
                        meta = SampleMetadata(
                            dataset_id=self.id,
                            sample_id=counter,
                            data={"path": member.name},
                        )
                        yield ImageSample(image=pil_image, meta=meta)

                        counter += 1
                        if counter % 2000 == 0:
                            logger.debug(f"Streamed {counter} images so far.")

                    except Exception as e:
                        logger.warning(f"Skipping corrupt file {member.name}: {e}")

        finally:
            # kill cat if the loop breaks early
            if process.poll() is None:
                process.kill()
            logger.info("Finished streaming images.")
