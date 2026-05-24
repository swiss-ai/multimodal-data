"""
Preprocess BigEarthNet.txt into a WebDataset for SFT.

Each sample contains:
  - {key}.png  : RGB (B04/B03/B02) PNG image at 240x240
  - {key}.json : multi-turn user/assistant conversation with all QA pairs
                 for that patch, plus metadata

Usage:
  # Write 10-sample preview to outputs/sample/
  python main.py --mode sample --n 10

  # Write full dataset to outputs/full_all/
  python main.py --mode full

  # Write one split only
  python main.py --mode full --split train
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import subprocess
import tarfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import webdataset as wds
from PIL import Image

try:
    import rasterio

    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PARQUET_PATH = (
    "/path/to/data/vision-datasets/raw/sft/hf___BIFOLD-BigEarthNetv2-0___BigEarthNet/text/BigEarthNet.txt.parquet"
)
S2_TAR_ZST = "/path/to/data/vision-datasets/raw/sft/hf___BIFOLD-BigEarthNetv2-0___BigEarthNet/BigEarthNet-S2.tar.zst"
PREPROCESSED_TARS = [
    "/path/to/data/vision-datasets/processed"
    "/hf___wangyi111___Copernicus-Bench___processed/bigearthnet"
    f"/bigearthnet-{i:06d}.tar"
    for i in range(3)
]
OUTPUT_DIR = Path(__file__).parent / "outputs"

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
CLIP_MAX = 2500
TARGET_SIZE = 240
SAMPLES_PER_SHARD = 1000
NUM_WORKERS = 128  # PNG conversion workers
QUEUE_DEPTH = 2048  # max buffered raw-band tuples waiting for workers
RESULT_DEPTH = 4096  # max buffered (patch_id, png, json) tuples waiting to be written
PZSTD_THREADS = 64  # parallel decompression threads


# ---------------------------------------------------------------------------
# RGB PNG conversion (runs in worker processes)
# ---------------------------------------------------------------------------
def _bands_to_png(b04: bytes, b03: bytes, b02: bytes) -> bytes:
    if not _HAS_RASTERIO:
        raise RuntimeError("rasterio required")
    bands = []
    for data in (b04, b03, b02):
        with rasterio.open(io.BytesIO(data)) as src:
            band = src.read(1).astype(np.float32)
            nd = src.nodata
        if nd is not None:
            band[band == nd] = np.nan
        bands.append(band)
    rgb = np.dstack(bands)
    rgb = np.clip(rgb, 0, CLIP_MAX)
    rgb = (rgb / CLIP_MAX * 255).astype(np.uint8)
    img = Image.fromarray(rgb).resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _worker(in_q: mp.Queue, out_q: mp.Queue):
    """Convert raw band bytes → PNG bytes in a worker process."""
    while True:
        item = in_q.get()
        if item is None:
            break
        patch_id, json_bytes, b04, b03, b02 = item
        try:
            png = _bands_to_png(b04, b03, b02)
            out_q.put((patch_id, png, json_bytes))
        except Exception as e:
            print(f"  ERROR processing {patch_id}: {e}")
    # Signal writer that this worker is done
    out_q.put(None)


# ---------------------------------------------------------------------------
# Parquet → in-memory lookup
# ---------------------------------------------------------------------------
def build_patch_lookup(split: str | None = None) -> dict[str, bytes]:
    """
    Load parquet and build {patch_id: json_bytes} for every patch.
    Keeps only patches belonging to `split` if given.
    All text data lives in RAM — easily fits in 700 GB.
    """
    print("Loading parquet …")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  {len(df):,} rows, {df['patch_id'].nunique():,} unique patches")

    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
        print(f"  After split={split} filter: {len(df):,} rows, {df['patch_id'].nunique():,} patches")

    print("Building per-patch conversation lookup …")
    lookup: dict[str, bytes] = {}
    for patch_id, grp in df.groupby("patch_id", sort=False):
        first = grp.iloc[0]
        meta = {
            "patch_id": patch_id,
            "latitude": float(first["latitude"]),
            "longitude": float(first["longitude"]),
            "country": first["country"],
            "season": first["season"],
            "climate_zone": first["climate_zone"],
        }
        turns = []
        for _, row in grp.iterrows():
            turns.append({"role": "user", "content": row["input"]})
            turns.append({"role": "assistant", "content": str(row["output"])})
        conv = {"conversations": turns, **meta}
        lookup[patch_id] = json.dumps(conv).encode()

    print(f"  Built lookup for {len(lookup):,} patches")
    return lookup


# ---------------------------------------------------------------------------
# Producer: stream tar.zst and push raw bands to the work queue
# ---------------------------------------------------------------------------
def _producer(in_q: mp.Queue, lookup: dict[str, bytes], num_workers: int):
    """
    Decompress BigEarthNet-S2.tar.zst with pzstd (parallel), stream the tar,
    collect B02/B03/B04 per patch, and push complete patches to in_q.
    Only patches present in `lookup` are pushed.
    """
    RGB_BANDS = {"B02", "B03", "B04"}
    wanted = set(lookup.keys())

    proc = subprocess.Popen(
        ["pzstd", "-d", "-c", f"-p{PZSTD_THREADS}", S2_TAR_ZST],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    current_patch: str | None = None
    band_data: dict[str, bytes] = {}
    pushed = 0

    def _push(pid, bdata):
        nonlocal pushed
        if pid in lookup and len(bdata) == 3:
            in_q.put((pid, lookup[pid], bdata["B04"], bdata["B03"], bdata["B02"]))
            pushed += 1
            if pushed % 10000 == 0:
                print(f"  [producer] {pushed:,} patches queued …", flush=True)

    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in tar:
                if not member.name.endswith(".tif"):
                    continue
                basename = os.path.basename(member.name)
                band = basename.rsplit("_", 1)[1].replace(".tif", "")
                patch_id = basename.rsplit("_", 1)[0]

                if patch_id != current_patch:
                    _push(current_patch, band_data)
                    current_patch = patch_id
                    band_data = {}

                if band in RGB_BANDS and patch_id in wanted:
                    f = tar.extractfile(member)
                    if f is not None:
                        band_data[band] = f.read()

        _push(current_patch, band_data)
    finally:
        proc.terminate()
        proc.wait()

    # Send sentinel values to shut down workers
    for _ in range(num_workers):
        in_q.put(None)
    print(f"  [producer] Done: {pushed:,} patches pushed", flush=True)


# ---------------------------------------------------------------------------
# Writer: drain result queue → webdataset shards
# ---------------------------------------------------------------------------
def _writer(out_q: mp.Queue, out_dir: Path, split_tag: str, total: int, num_workers: int):
    pattern = str(out_dir / f"bigearthnet-{split_tag}-%06d.tar")
    sink = wds.ShardWriter(pattern, maxcount=SAMPLES_PER_SHARD)
    written = 0
    done_workers = 0

    while done_workers < num_workers:
        try:
            item = out_q.get(timeout=5)
        except Exception:
            continue
        if item is None:
            done_workers += 1
            continue
        patch_id, png, json_bytes = item
        sink.write({"__key__": patch_id, "png": png, "json": json_bytes})
        written += 1
        if written % 5000 == 0:
            print(f"  [writer] {written:,} / {total:,} written …", flush=True)

    sink.close()
    print(f"  [writer] Done: {written:,} samples → {out_dir}", flush=True)


# ---------------------------------------------------------------------------
# Full parallel pipeline
# ---------------------------------------------------------------------------
def write_full(split: str | None = None, out_dir: Path | None = None):
    split_tag = split if split else "all"
    out_dir = out_dir or (OUTPUT_DIR / f"full_{split_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    lookup = build_patch_lookup(split)
    total = len(lookup)

    # Use spawn context to avoid fork issues with rasterio/gdal
    ctx = mp.get_context("spawn")
    in_q = ctx.Queue(maxsize=QUEUE_DEPTH)
    out_q = ctx.Queue(maxsize=RESULT_DEPTH)

    # Start worker processes
    workers = [ctx.Process(target=_worker, args=(in_q, out_q), daemon=True) for _ in range(NUM_WORKERS)]
    for w in workers:
        w.start()

    # Writer runs in the main thread
    writer_thread = threading.Thread(
        target=_writer,
        args=(out_q, out_dir, split_tag, total, NUM_WORKERS),
        daemon=False,
    )
    writer_thread.start()

    # Producer runs in the main thread (streaming tar — must be single process)
    _producer(in_q, lookup, NUM_WORKERS)

    writer_thread.join()
    print(f"\nAll done. Output: {out_dir}")


# ---------------------------------------------------------------------------
# Sample mode (unchanged — uses preprocessed PNGs, no multiprocessing needed)
# ---------------------------------------------------------------------------
def write_sample(n: int = 10):
    out_dir = OUTPUT_DIR / "sample"
    out_dir.mkdir(parents=True, exist_ok=True)

    lookup = build_patch_lookup()

    # Pick from patches that already have preprocessed PNGs
    preprocessed = set()
    for tp in PREPROCESSED_TARS:
        if os.path.exists(tp):
            with tarfile.open(tp) as t:
                for m in t.getmembers():
                    preprocessed.add(Path(m.name).stem)

    available = [p for p in preprocessed if p in lookup]
    chosen = available[:n]
    print(f"Selected {len(chosen)} patches from preprocessed tars")

    # Load PNGs
    png_map: dict[str, bytes] = {}
    wanted = set(chosen)
    for tp in PREPROCESSED_TARS:
        if not wanted or not os.path.exists(tp):
            continue
        with tarfile.open(tp) as t:
            for m in t.getmembers():
                key = Path(m.name).stem
                if key in wanted:
                    f = t.extractfile(m)
                    if f:
                        png_map[key] = f.read()
                        wanted.discard(key)

    pattern = str(out_dir / "sample-%06d.tar")
    sink = wds.ShardWriter(pattern, maxcount=SAMPLES_PER_SHARD)
    for patch_id in chosen:
        if patch_id not in png_map:
            continue
        sink.write({"__key__": patch_id, "png": png_map[patch_id], "json": lookup[patch_id]})
        print(f"  Wrote {patch_id}")
    sink.close()
    print(f"\nSample written: {len(chosen)} samples → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BigEarthNet.txt → WebDataset")
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--n", type=int, default=10, help="Patches for sample mode")
    parser.add_argument(
        "--split",
        default=None,
        choices=["train", "validation", "test", "bench"],
        help="Filter to split (full mode only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help="PNG conversion worker processes",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory (full mode only)",
    )
    args = parser.parse_args()

    if args.mode == "sample":
        write_sample(n=args.n)
    else:
        write_full(split=args.split, out_dir=args.out_dir)
