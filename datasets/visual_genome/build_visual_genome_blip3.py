"""Convert Visual Genome → BLIP3-Grounding Level-2 dense-caption parquet.

Output one row per image, format:
    <object>phrase_1</object><bbox>[x1, y1][x2, y2]</bbox>,
    <object>phrase_2</object><bbox>[x1, y1][x2, y2]</bbox>, …

All bboxes are human-drawn ground truth from VG (renormalized px-xywh → 0-1 xyxy).
No metadata prefix, no opener prose, no Qwen — pure data transformation.

Output schema (per shard):
    image_id: int64
    image: struct<bytes: binary, path: string>
    width: int32
    height: int32
    n_regions: int32
    grounded_text: string   ← BLIP3 Level-2 dense list

Output dir:
    raw/stage2/visual_genome/parquet_blip3/part-NNNNN-of-NNNNN.parquet
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ─── schema ──────────────────────────────────────────────────────────────────

SCHEMA = pa.schema([
    ("image_id",      pa.int64()),
    ("image",         pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("width",         pa.int32()),
    ("height",        pa.int32()),
    ("n_regions",     pa.int32()),
    ("grounded_text", pa.string()),
])


def norm_bbox_xywh(x, y, w, h, cw, ch):
    if cw <= 0 or ch <= 0: return None
    x1 = max(0.0, x / cw); y1 = max(0.0, y / ch)
    x2 = min(1.0, (x + w) / cw); y2 = min(1.0, (y + h) / ch)
    if x2 <= x1 + 1e-3 or y2 <= y1 + 1e-3: return None
    return x1, y1, x2, y2


def fmt_grounded_text(regions, cw, ch):
    parts = []
    for r in regions:
        b = norm_bbox_xywh(r['x'], r['y'], r['width'], r['height'], cw, ch)
        if b is None: continue
        phrase = (r['phrase'] or '').strip().rstrip('.').replace('\n', ' ')
        if not phrase: continue
        x1, y1, x2, y2 = b
        parts.append(
            f"<object>{phrase}</object>"
            f"<bbox>[{x1:.3f}, {y1:.3f}][{x2:.3f}, {y2:.3f}]</bbox>"
        )
    return ", ".join(parts) if parts else ""


def find_jpeg(image_id: int, image_dirs):
    for d in image_dirs:
        p = d / f"{image_id}.jpg"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/stage2/visual_genome")
    ap.add_argument("--output", default="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/stage2/visual_genome/parquet_blip3")
    ap.add_argument("--shard-rows", type=int, default=2000)
    ap.add_argument("--read-workers", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ann = root / "annotations"
    image_dirs = [
        root / "images",            # Part 1 extracted flat
        root / "images" / "VG_100K_2",  # Part 2 extracted into subdir
    ]

    t0 = time.time()
    print(f'[{time.time()-t0:5.1f}s] loading image_data', flush=True)
    img_data = {d['image_id']: d for d in json.load(open(ann/'image_data.json'))}
    print(f'           → {len(img_data):,} images')

    print(f'[{time.time()-t0:5.1f}s] loading region_descriptions', flush=True)
    regions_by_img = {d['id']: d['regions'] for d in json.load(open(ann/'region_descriptions.json'))}
    print(f'           → regions for {len(regions_by_img):,} images')

    image_ids = sorted(img_data.keys())
    n = len(image_ids)
    n_shards = (n + args.shard_rows - 1) // args.shard_rows
    print(f'\n[{time.time()-t0:5.1f}s] writing {n_shards} shards × ~{args.shard_rows} rows = {n:,} images')

    def build_row(img_id):
        meta = img_data[img_id]
        cw = int(meta.get('width') or 0); ch = int(meta.get('height') or 0)
        regions = regions_by_img.get(img_id, [])
        text = fmt_grounded_text(regions, cw, ch)
        if not text:
            return None
        jpeg_path = find_jpeg(img_id, image_dirs)
        if jpeg_path is None:
            return None
        try:
            jpeg_bytes = jpeg_path.read_bytes()
        except Exception:
            return None
        return {
            "image_id": int(img_id),
            "image": {"bytes": jpeg_bytes, "path": jpeg_path.name},
            "width": cw,
            "height": ch,
            "n_regions": len(regions),
            "grounded_text": text,
        }

    n_written = 0
    n_skipped = 0
    bytes_written = 0
    with ThreadPoolExecutor(max_workers=args.read_workers) as ex:
        for shard_idx in range(n_shards):
            lo = shard_idx * args.shard_rows
            hi = min(lo + args.shard_rows, n)
            ids = image_ids[lo:hi]
            rows = list(filter(None, ex.map(build_row, ids)))
            n_skipped += len(ids) - len(rows)
            if not rows: continue
            out_path = out_dir / f"part-{shard_idx:05d}-of-{n_shards:05d}.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out_path,
                           compression="zstd")
            sz = out_path.stat().st_size
            bytes_written += sz
            n_written += len(rows)
            elapsed = time.time() - t0
            print(f'  [{elapsed:6.1f}s] shard {shard_idx+1}/{n_shards}  '
                  f'rows={n_written:,}  size={bytes_written/1e9:.2f} GB', flush=True)

    elapsed = time.time() - t0
    print(f'\n=== DONE in {elapsed:.0f}s ({elapsed/60:.1f} min) ===')
    print(f'  rows written:  {n_written:,}')
    print(f'  rows skipped:  {n_skipped} (missing image, empty regions, or zero dims)')
    print(f'  total parquet: {bytes_written/1e9:.2f} GB across {n_shards} shards')
    print(f'  output:        {out_dir}')


if __name__ == "__main__":
    main()
