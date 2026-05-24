import io
import os
import zipfile
from pathlib import Path

import webdataset as wds
from PIL import Image

data_dir = Path(os.getenv("TIFF_TO_JPEG_DATA_DIR", "/path/to/data"))
output_dir = Path(os.getenv("TIFF_TO_JPEG_OUTPUT_DIR", "/path/to/output"))


def tiff_to_jpeg(tiff_bytes: io.BytesIO) -> bytes:
    img = Image.open(tiff_bytes)
    img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=100)
    return out.getvalue()


def main():
    # create a writer
    ds = wds.ShardWriter(str(output_dir / "data-%06d.tar"), maxcount=100000)
    for zip_path in sorted(data_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                assert member.lower().endswith(".tif")
                with zf.open(member) as fh:
                    img_bytes = fh.read()
                    # convert to JPEG (get rid of alpha channel)
                    img_bytes = tiff_to_jpeg(io.BytesIO(img_bytes))
                    key = f"{zip_path.stem}___{member.replace('/', '__').replace('.tif', '')}"
                    ds.write({"__key__": key, "jpg": img_bytes})
    ds.close()


if __name__ == "__main__":
    main()
