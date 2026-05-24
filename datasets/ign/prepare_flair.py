import argparse
import io
import zipfile
from pathlib import Path

import webdataset as wds
from PIL import Image


def tiff_to_jpeg(tiff_bytes: io.BytesIO) -> bytes:
    img = Image.open(tiff_bytes)
    img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=100)
    return out.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Convert IGN FLAIR-HUB TIFFs to WDS JPEG shards")
    parser.add_argument("--input-dir", required=True, help="Path to FLAIR-HUB data directory")
    parser.add_argument("--output-dir", required=True, help="Path to output WDS shards")
    parser.add_argument("--maxcount", type=int, default=100000, help="Samples per shard")
    args = parser.parse_args()

    data_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = wds.ShardWriter(str(output_dir / "data-%06d.tar"), maxcount=args.maxcount)
    for zip_path in sorted(data_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                assert member.lower().endswith(".tif")
                with zf.open(member) as fh:
                    img_bytes = fh.read()
                    img_bytes = tiff_to_jpeg(io.BytesIO(img_bytes))
                    key = f"{zip_path.stem}___{member.replace('/', '__').replace('.tif', '')}"
                    ds.write({"__key__": key, "jpg": img_bytes})
    ds.close()


if __name__ == "__main__":
    main()
