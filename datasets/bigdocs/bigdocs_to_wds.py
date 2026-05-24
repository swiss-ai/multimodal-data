import argparse
import glob
import os
import tarfile
import zipfile

import pyarrow.parquet as pq
import webdataset as wds

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=[
        "arxiv_ocr",
        "arxiv_tablecap",
        "pubtables1m",
        "cocotext",
        "textocr",
        "cord_v2",
    ],
)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--max-per-shard", type=int, default=10000)
args = parser.parse_args()


# ==========================================
# CONFIGURATION
# ==========================================

# fmt:off
COCOTEXT_ZIP = "/tmp/bigdocs_subsets/train2014.zip"
TEXTOCR_ZIP = "/tmp/bigdocs_subsets/train_val_images.zip"
BIGDOCS_SNAPSHOT = "/path/to/data/vision-datasets/hf_hub_cache/datasets--ServiceNow--BigDocs-7.5M/snapshots/dae4403c28307bd5328920740e81ce5232819e74"
PUBTABLES_SNAPSHOT = "/path/to/data/vision-datasets/hf_hub_cache/datasets--bsmock--pubtables-1m/snapshots/35b1c097807e0b07ec5313879b85956b7b3890db"
PUBTABLES_IMAGE_ARCHIVES = [
    "PubTables-1M-Detection_Images_Train_Part1.tar.gz",
    "PubTables-1M-Detection_Images_Train_Part2.tar.gz",
    "PubTables-1M-Detection_Images_Val.tar.gz",
    "PubTables-1M-Detection_Images_Test.tar.gz",
    "PubTables-1M-Structure_Images_Train.tar.gz",
    "PubTables-1M-Structure_Images_Val.tar.gz",
    "PubTables-1M-Structure_Images_Test.tar.gz",
]
# fmt:on


# ==========================================
# DATASET LOADERS
# ==========================================
# Each loader yields (sample_key: str, image_bytes: bytes) pairs.


def _load_bigdocs_image_subset(subset_name):
    """Load images from a BigDocs-7.5M subset that has embedded image bytes.

    Reads parquet files from the HF cache, extracts image.bytes (PNG) per row.
    """
    subset_dir = os.path.join(BIGDOCS_SNAPSHOT, subset_name)
    parquet_files = sorted(glob.glob(os.path.join(subset_dir, "train-*.parquet")))
    print(f"[{subset_name}] Found {len(parquet_files)} train parquet files")

    for pi, pf in enumerate(parquet_files):
        table = pq.read_table(pf, columns=["sample_id", "image"])
        sample_ids = table.column("sample_id")
        images = table.column("image")

        for i in range(len(table)):
            key = sample_ids[i].as_py().replace(".", "_")
            img_struct = images[i].as_py()
            img_bytes = img_struct.get("bytes")
            if img_bytes is None:
                continue
            yield key, img_bytes

        if (pi + 1) % 50 == 0:
            print(f"[{subset_name}] Processed {pi + 1}/{len(parquet_files)} parquets")


def load_arxiv_ocr():
    """Load ArxivOCR images from BigDocs-7.5M parquets (1127 train files, PNG)."""
    return _load_bigdocs_image_subset("ArxivOCR")


def load_arxiv_tablecap():
    """Load ArxivTableCap images from BigDocs-7.5M parquets (12 train files, PNG)."""
    return _load_bigdocs_image_subset("ArxivTableCap")


def load_cord_v2():
    """Load cord-v2 images from BigDocs-7.5M parquets (PNG)."""
    return _load_bigdocs_image_subset("cord-v2")


def load_pubtables1m():
    """Load pubtables-1m images by streaming from tar.gz archives in HF cache.

    Images appear in both Detection and Structure archives with the same
    filenames (e.g. PMC4683861_3.jpg), so we deduplicate by key.
    """
    seen = set()
    for archive_name in PUBTABLES_IMAGE_ARCHIVES:
        archive_path = os.path.join(PUBTABLES_SNAPSHOT, archive_name)
        print(f"[pubtables1m] Streaming from {archive_name}...")
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf:
                if not member.isfile() or not member.name.lower().endswith(".jpg"):
                    continue
                key = os.path.splitext(os.path.basename(member.name))[0]
                if key in seen:
                    continue
                seen.add(key)
                f = tf.extractfile(member)
                if f is None:
                    continue
                img_bytes = f.read()
                yield key, img_bytes


def load_cocotext():
    """Load COCOtext images from train2014.zip."""
    with zipfile.ZipFile(COCOTEXT_ZIP, "r") as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith(".jpg"):
                continue
            # key: e.g. "COCO_train2014_000000270070"
            key = os.path.splitext(os.path.basename(name))[0]
            img_bytes = zf.read(name)
            yield key, img_bytes


def load_textocr():
    """Load TextOCR images from train_val_images.zip."""
    with zipfile.ZipFile(TEXTOCR_ZIP, "r") as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith(".jpg"):
                continue
            # key: e.g. "03b2dbb7dff0e32d"
            key = os.path.splitext(os.path.basename(name))[0]
            img_bytes = zf.read(name)
            yield key, img_bytes


LOADERS = {
    "arxiv_ocr": load_arxiv_ocr,
    "arxiv_tablecap": load_arxiv_tablecap,
    "pubtables1m": load_pubtables1m,
    "cord_v2": load_cord_v2,
    "cocotext": load_cocotext,
    "textocr": load_textocr,
}


# ==========================================
# WRITER
# ==========================================


def write_webdataset(dataset_name, loader_fn, output_dir, max_per_shard):
    ds_output = os.path.join(output_dir, dataset_name)
    os.makedirs(ds_output, exist_ok=True)

    pattern = os.path.join(ds_output, "part-%06d.tar")
    sink = wds.ShardWriter(pattern, maxcount=max_per_shard)

    total = 0
    for key, img_bytes in loader_fn():
        if img_bytes[:4] == b"\x89PNG":
            sink.write({"__key__": key, "png": img_bytes})
        elif img_bytes[:3] == b"\xff\xd8\xff":
            sink.write({"__key__": key, "jpg": img_bytes})
        else:
            raise ValueError(f"Unsupported image format for key {key}")
        total += 1
        if total % 5000 == 0:
            print(f"[{dataset_name}] Written {total} samples...")

    sink.close()
    print(f"[{dataset_name}] Done. Total: {total} samples")


# ==========================================
# MAIN
# ==========================================


def main():
    print(f"Dataset: {args.dataset}")
    print(f"Output:  {args.output_dir}")
    print(f"Max/shard: {args.max_per_shard}")

    loader_fn = LOADERS[args.dataset]
    write_webdataset(args.dataset, loader_fn, args.output_dir, args.max_per_shard)


if __name__ == "__main__":
    main()
