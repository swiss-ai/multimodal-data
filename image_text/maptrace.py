import hashlib
import os

import pyarrow.parquet as pq
import webdataset

BASE = "/path/to/data/vision-datasets/hf_hub_cache/datasets--google--MapTrace/snapshots/00bae0d2d917fd12548a089285d633dadf1bc81c"
DATASETS = {
    "maptrace": os.path.join(BASE, "maptrace"),
    "floormaps": os.path.join(BASE, "floormaps"),
}
OUT_DIR = "/path/to/data/vision-datasets/MapTrace"


def detect_ext(image_bytes):
    if image_bytes[:4] == b"\x89PNG":
        return "png"
    elif image_bytes[:2] == b"\xff\xd8":
        return "jpg"
    elif image_bytes[:4] in (b"GIF8", b"GIF9"):
        return "gif"
    elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "webp"
    else:
        raise ValueError("Unknown image format")


def embed_placeholders(text, n_images):
    placeholders = [f"<|img{i}|>" for i in range(1, n_images + 1)]
    missing = [p for p in placeholders if p not in text]
    if missing:
        text = "\n".join(missing) + "\n" + text
    return text


pattern = os.path.join(OUT_DIR, "part-%06d.tar")
os.makedirs(OUT_DIR, exist_ok=True)
sink = webdataset.ShardWriter(pattern, maxcount=10000)

seen_hashes = set()
txt_hashes = set()
total_written = 0
total_skipped = 0

for dataset_name, dataset_dir in DATASETS.items():
    files = sorted(f for f in os.listdir(dataset_dir) if f.endswith(".parquet"))
    print(f"Processing {dataset_name}: {len(files)} parquet files")

    for i, fname in enumerate(files):
        table = pq.read_table(os.path.join(dataset_dir, fname))
        for row_idx in range(len(table)):
            image_bytes = table["image_bytes"][row_idx].as_py()
            caption = table["map_description"][row_idx].as_py()

            img_hash = hashlib.md5(image_bytes).hexdigest()
            txt_hash = hashlib.md5(caption.encode("utf-8")).hexdigest()

            if img_hash in seen_hashes:
                assert txt_hash in txt_hashes
                total_skipped += 1
                continue
            seen_hashes.add(img_hash)
            txt_hashes.add(txt_hash)

            ext = detect_ext(image_bytes)
            assert ext == "png"
            text = embed_placeholders(caption, n_images=1)

            sink.write(
                {
                    "__key__": f"{dataset_name}__{img_hash}",
                    "img1.png": image_bytes,
                    "txt": text.encode("utf-8"),
                }
            )
            total_written += 1

        if (i + 1) % 100 == 0:
            print(f"  [{dataset_name}] {i + 1}/{len(files)} files, written={total_written}, skipped={total_skipped}")

    print(f"  Done {dataset_name}: written={total_written}, skipped={total_skipped}")

sink.close()
print(f"\nFinished. Unique pairs: {total_written}, duplicates: {total_skipped}")
