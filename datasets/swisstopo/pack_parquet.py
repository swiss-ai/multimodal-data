#!/usr/bin/env python3
"""pack_parquet.py — Pack swisstopo captioned tiles into parquet.

For each shard:
  - Load outputs/captions_full/shard_NNNNN.jsonl into a dict keyed by `key`.
  - Stream the corresponding tar at DATA_ROOT/NNNNN.tar once.
  - For each .png whose stem is in the caption dict, also read its sibling
    .json (full original metadata) and emit a row.

Output: one parquet file per shard (split into parts if > MAX_BYTES).
"""

import argparse
import json
import multiprocessing as mp
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATA_ROOT = Path(os.environ.get("SWISSTOPO_DATA_ROOT", ""))
CAPTIONS_ROOT = Path(os.environ.get("SWISSTOPO_CAPTIONS_ROOT", "/tmp/toolbox/swisstopo_maps/outputs/captions_full"))
OUTPUT_ROOT_FULL = Path(os.environ.get("SWISSTOPO_OUTPUT_FULL", ""))
OUTPUT_ROOT_SAMPLE = Path(os.environ.get("SWISSTOPO_OUTPUT_SAMPLE", "/tmp/toolbox/swisstopo_maps/outputs/sample_parquet"))
NUM_SHARDS = 154
MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
ROW_GROUP_SIZE = 256
FLUSH_BYTES = 256 * 1024 * 1024

SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("image_bytes", pa.binary()),
        ("caption", pa.string()),
        ("layer", pa.string()),
        ("scale", pa.float64()),
        ("img_w", pa.int32()),
        ("lang", pa.string()),
        ("building_frac", pa.float64()),
        ("bbox", pa.string()),
        ("metadata", pa.string()),
    ]
)


def load_captions(shard_idx: int) -> dict[str, dict]:
    path = CAPTIONS_ROOT / f"shard_{shard_idx:05d}.jsonl"
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["key"]] = r
    return out


def iter_rows(shard_idx: int, captions: dict[str, dict], limit: int | None = None):
    tar_path = DATA_ROOT / f"{shard_idx:05d}.tar"
    pending: dict[str, object] = {}  # key -> png_bytes OR parsed-json dict
    emitted = 0
    with tarfile.open(tar_path, "r|") as tf:  # streaming mode
        for member in tf:
            if not member.isfile():
                continue
            stem, _, ext = member.name.rpartition(".")
            if stem not in captions:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            data = f.read()
            if ext == "png":
                prev = pending.get(stem)
                if isinstance(prev, dict):
                    pending.pop(stem)
                    yield _make_row(stem, data, prev, captions[stem])
                    emitted += 1
                else:
                    pending[stem] = data
            elif ext == "json":
                meta = json.loads(data.decode("utf-8"))
                prev = pending.get(stem)
                if isinstance(prev, (bytes, bytearray)):
                    pending.pop(stem)
                    yield _make_row(stem, bytes(prev), meta, captions[stem])
                    emitted += 1
                else:
                    pending[stem] = meta
            if limit is not None and emitted >= limit:
                return


def _make_row(key: str, png: bytes, meta: dict, cap: dict) -> dict:
    full_meta = dict(meta)
    full_meta["shard"] = cap["shard"]
    full_meta["key"] = key
    return {
        "sample_id": meta.get("sample_id", ""),
        "image_bytes": png,
        "caption": cap["caption"],
        "layer": cap["layer"],
        "scale": float(cap["scale"]),
        "img_w": int(cap["img_w"]),
        "lang": cap["lang"],
        "building_frac": float(cap["building_frac"]),
        "bbox": meta.get("bbox", ""),
        "metadata": json.dumps(full_meta, ensure_ascii=False),
    }


def _row_size(r: dict) -> int:
    return (
        len(r["sample_id"])
        + len(r["image_bytes"])
        + len(r["caption"])
        + len(r["layer"])
        + len(r["lang"])
        + len(r["bbox"])
        + len(r["metadata"])
        + 32
    )


def pack_shard(shard_idx: int, out_root: Path, limit: int | None = None) -> tuple[int, int]:
    out_root.mkdir(parents=True, exist_ok=True)
    captions = load_captions(shard_idx)
    if not captions:
        return (0, 0)

    part_idx = 0
    writer: pq.ParquetWriter | None = None
    out_path: Path | None = None
    bytes_written = 0
    rows_buf: list[dict] = []
    bytes_buf = 0
    total = 0

    def open_writer():
        nonlocal part_idx
        path = out_root / f"shard_{shard_idx:05d}-part_{part_idx:03d}.parquet"
        return pq.ParquetWriter(path, SCHEMA, compression="zstd"), path

    def flush(w: pq.ParquetWriter):
        nonlocal rows_buf, bytes_buf, bytes_written
        if not rows_buf:
            return
        cols = {n: [r[n] for r in rows_buf] for n in SCHEMA.names}
        w.write_table(pa.table(cols, schema=SCHEMA), row_group_size=ROW_GROUP_SIZE)
        bytes_written += bytes_buf
        rows_buf = []
        bytes_buf = 0

    for row in iter_rows(shard_idx, captions, limit=limit):
        rsz = _row_size(row)
        if writer is None:
            writer, out_path = open_writer()
            bytes_written = 0
        if bytes_written + bytes_buf + rsz > MAX_BYTES and (bytes_written + bytes_buf) > 0:
            flush(writer)
            writer.close()
            part_idx += 1
            writer, out_path = open_writer()
            bytes_written = 0
        rows_buf.append(row)
        bytes_buf += rsz
        total += 1
        if bytes_buf >= FLUSH_BYTES:
            flush(writer)

    if writer is not None:
        flush(writer)
        writer.close()

    return (total, part_idx + 1 if writer is not None else 0)


def _worker(args):
    shard_idx, out_root_str = args
    out_root = Path(out_root_str)
    try:
        n, parts = pack_shard(shard_idx, out_root)
        print(f"shard {shard_idx:05d}: {n} rows -> {parts} part(s)", flush=True)
        return (shard_idx, n, parts, None)
    except Exception as e:
        print(f"shard {shard_idx:05d}: ERROR {e}", flush=True)
        return (shard_idx, 0, 0, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "shard", "full"], required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n", type=int, default=5, help="sample mode: rows to write")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "sample":
        out = Path(args.out) if args.out else OUTPUT_ROOT_SAMPLE
        out.mkdir(parents=True, exist_ok=True)
        n, parts = pack_shard(args.shard, out, limit=args.n)
        print(f"sample: shard {args.shard} -> {n} rows in {parts} file(s) at {out}")
        return

    if args.mode == "shard":
        out = Path(args.out) if args.out else OUTPUT_ROOT_FULL
        n, parts = pack_shard(args.shard, out)
        print(f"shard {args.shard}: {n} rows -> {parts} part(s)")
        return

    # full
    out = Path(args.out) if args.out else OUTPUT_ROOT_FULL
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(s, str(out)) for s in range(NUM_SHARDS)]
    print(f"Packing {NUM_SHARDS} shards with {args.workers} workers -> {out}")
    total_rows = 0
    errors = []
    with mp.Pool(args.workers) as pool:
        for shard_idx, n, parts, err in pool.imap_unordered(_worker, tasks):
            total_rows += n
            if err:
                errors.append((shard_idx, err))
    print(f"DONE. total rows: {total_rows}. errors: {len(errors)}")
    for s, e in errors:
        print(f"  shard {s}: {e}")


if __name__ == "__main__":
    main()
