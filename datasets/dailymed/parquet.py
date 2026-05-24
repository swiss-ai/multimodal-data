"""
DailyMed SPL: unpack nested zips into flat parquet shards.

Each top-level zip in SRC_DIR contains many inner zips, each holding 1 SPL XML
file plus 0..N images. This script flattens the corpus to parquet rows of
shape (id, xml, images) where images is a list<struct{name, bytes}>.

Run interactively:
    .venv/bin/python parquet.py
"""

import io
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SRC_DIR = "/path/to/data/medical-datasets/raw/dailymed_spl/raw_zips2"
DST_DIR = "/path/to/data/medical-datasets/raw/dailymed_spl/parquet"
NUM_WORKERS = 256
SHARD_TARGET_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
ROW_GROUP_ROWS = 256
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 3

# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------
IMAGE_STRUCT = pa.struct(
    [
        pa.field("name", pa.string()),
        pa.field("bytes", pa.binary()),
    ]
)
SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("xml", pa.string()),
        pa.field("images", pa.list_(IMAGE_STRUCT)),
    ]
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".svg")


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------
def category_from_outer_name(outer_basename: str) -> str:
    # dm_spl_release_human_otc_part3.zip -> human_otc_part3
    name = outer_basename
    if name.endswith(".zip"):
        name = name[:-4]
    prefix = "dm_spl_release_"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    return name


def list_inner_zips(outer_path: str):
    """Return list of (outer_path, inner_name) for every inner .zip entry."""
    out = []
    with zipfile.ZipFile(outer_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if info.filename.lower().endswith(".zip"):
                out.append((outer_path, info.filename))
    return out


def process_inner(task):
    """Worker: extract one inner zip's bytes from outer, parse, return row dict."""
    outer_path, inner_name = task
    category = category_from_outer_name(os.path.basename(outer_path))
    inner_basename = os.path.basename(inner_name)
    if inner_basename.lower().endswith(".zip"):
        inner_basename = inner_basename[:-4]
    row_id = f"{category}/{inner_basename}"

    try:
        with zipfile.ZipFile(outer_path) as outer:
            with outer.open(inner_name) as f:
                inner_bytes = f.read()

        xml_text = ""
        images = []
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
            for info in inner.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                lower = name.lower()
                data = inner.read(info)
                if lower.endswith(".xml"):
                    # Decode permissively; SPL is declared UTF-8.
                    xml_text = data.decode("utf-8", errors="replace")
                elif lower.endswith(IMAGE_EXTS):
                    images.append({"name": os.path.basename(name), "bytes": data})
                # silently ignore anything else (rare)

        return {"id": row_id, "xml": xml_text, "images": images}
    except Exception as e:
        return {"_error": f"{row_id}: {type(e).__name__}: {e}"}


# -----------------------------------------------------------------------------
# Writer
# -----------------------------------------------------------------------------
class ShardWriter:
    def __init__(self, dst_dir: str, target_bytes: int):
        self.dst_dir = dst_dir
        self.target_bytes = target_bytes
        self.shard_idx = 0
        self.writer = None
        self.current_path = None
        os.makedirs(dst_dir, exist_ok=True)

    def _open_new(self):
        self.current_path = os.path.join(self.dst_dir, f"part-{self.shard_idx:05d}.parquet")
        self.writer = pq.ParquetWriter(
            self.current_path,
            SCHEMA,
            compression=COMPRESSION,
            compression_level=COMPRESSION_LEVEL,
        )

    def write_batch(self, rows):
        if not rows:
            return
        if self.writer is None:
            self._open_new()
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        self.writer.write_table(table, row_group_size=ROW_GROUP_ROWS)
        # Rotate by file size on disk.
        try:
            sz = os.path.getsize(self.current_path)
        except OSError:
            sz = 0
        if sz >= self.target_bytes:
            self.writer.close()
            self.writer = None
            self.shard_idx += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    outer_zips = sorted(os.path.join(SRC_DIR, n) for n in os.listdir(SRC_DIR) if n.lower().endswith(".zip"))
    print(f"[enum] {len(outer_zips)} outer zips in {SRC_DIR}", flush=True)

    t0 = time.time()
    tasks = []
    for op in outer_zips:
        sub = list_inner_zips(op)
        tasks.extend(sub)
        print(f"[enum]   {os.path.basename(op)}: {len(sub)} inner zips", flush=True)
    print(f"[enum] total {len(tasks)} inner zips in {time.time() - t0:.1f}s", flush=True)

    writer = ShardWriter(DST_DIR, SHARD_TARGET_BYTES)
    batch = []
    BATCH_ROWS = 256
    n_ok = 0
    n_err = 0
    err_log = open(os.path.join(DST_DIR, "_errors.log"), "w")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = [ex.submit(process_inner, t) for t in tasks]
        pbar = tqdm(total=len(futures), unit="doc", smoothing=0.05)
        for fut in as_completed(futures):
            row = fut.result()
            if "_error" in row:
                n_err += 1
                err_log.write(row["_error"] + "\n")
            else:
                batch.append(row)
                n_ok += 1
                if len(batch) >= BATCH_ROWS:
                    writer.write_batch(batch)
                    batch = []
            pbar.update(1)
        pbar.close()

    if batch:
        writer.write_batch(batch)
    writer.close()
    err_log.close()

    print(f"[done] {n_ok} ok, {n_err} errors. shards in {DST_DIR}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
