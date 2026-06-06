"""
prep.py: convert MolmoPoint-GUISyn parquet rows into SFT chat format with
0-1000 normalized coordinates.

ONE sample per source image, all annotations packed as multi-turn pairs
(Molmo's "message-trees" approach). seqlen-based stage2/lct routing happens
at tokenize time via the tokenize config's `seqlen_threshold` — we don't chunk.

Input: /capstor/.../raw/sft/hf___allenai___MolmoPoint-GUISyn/{desktop,web,mobile}/*.parquet
   row fields: id, image{bytes,path}, annotation[{x_center,y_center,width,height,name,intent[]}]

Output: /capstor/.../processed/sft/molmopoint_guisyn/{desktop,web,mobile}/*.parquet
   row fields: image (bytes), messages (list[{role,content}]), source, split

Per-sample format (multi-turn, one image, all annotations):
   [user]      <image>\n{intent_1}
   [assistant] <point x="{cx_1}" y="{cy_1}">{name_1}</point>
   [user]      {intent_2}
   [assistant] <point x="{cx_2}" y="{cy_2}">{name_2}</point>
   ... (one turn pair per annotation, intent picked deterministically per --seed)
"""

import argparse
import glob
import hashlib
import io
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def _deterministic_file_seed(global_seed: int, file_path: str) -> int:
    """Per-file seed derived deterministically (Python's hash() is randomized
    across processes unless PYTHONHASHSEED is set, so we use MD5 instead)."""
    h = hashlib.md5(os.path.basename(file_path).encode("utf-8")).digest()
    return (global_seed + int.from_bytes(h[:8], "big")) % (2**63)


def normalize_xy(x_center, y_center, W, H):
    cx = max(0, min(1000, round(x_center / W * 1000)))
    cy = max(0, min(1000, round(y_center / H * 1000)))
    return cx, cy


def build_sample(row, rng, split):
    """Pack one image + all its annotations into one multi-turn sample."""
    img_bytes = row["image"]["bytes"]
    try:
        W, H = Image.open(io.BytesIO(img_bytes)).size
    except Exception:
        return None
    if W <= 0 or H <= 0:
        return None

    msgs = []
    for i, ann in enumerate(row["annotation"] or []):
        intents = ann.get("intent") or []
        if not intents:
            continue
        intent = rng.choice(intents)
        name = (ann.get("name") or "").strip()
        cx, cy = normalize_xy(ann["x_center"], ann["y_center"], W, H)
        # First user turn includes <image>\n prefix; subsequent turns don't
        user_content = f"<image>\n{intent}" if not msgs else intent
        msgs.append({"role": "user", "content": user_content})
        msgs.append({"role": "assistant",
                     "content": f'<point x="{cx}" y="{cy}">{name}</point>'})

    if not msgs:
        return None
    return {
        "image": img_bytes,
        "messages": msgs,
        "source": row["id"],
        "split": split,
    }


def _flush(writer, buf, schema):
    if not buf["image"]:
        return 0
    batch = pa.table(buf, schema=schema)
    writer.write_table(batch)
    n = len(buf["image"])
    for k in buf:
        buf[k].clear()
    return n


def process_file(in_path, out_path, seed, split, batch_size):
    rng = random.Random(_deterministic_file_seed(seed, in_path))
    pf = pq.ParquetFile(in_path)
    schema = pa.schema([
        ("image", pa.binary()),
        ("messages", pa.list_(pa.struct([
            ("role", pa.string()),
            ("content", pa.string()),
        ]))),
        ("source", pa.string()),
        ("split", pa.string()),
    ])
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
    buf = {"image": [], "messages": [], "source": [], "split": []}
    total = 0
    n_rows_total = 0
    for batch in pf.iter_batches(batch_size=64, columns=["id", "image", "annotation"]):
        for i in range(batch.num_rows):
            row = {
                "id": batch["id"][i].as_py(),
                "image": batch["image"][i].as_py(),
                "annotation": batch["annotation"][i].as_py(),
            }
            n_rows_total += 1
            s = build_sample(row, rng, split)
            if s is None:
                continue
            buf["image"].append(s["image"])
            buf["messages"].append(s["messages"])
            buf["source"].append(s["source"])
            buf["split"].append(s["split"])
            if len(buf["image"]) >= batch_size:
                total += _flush(writer, buf, schema)
    total += _flush(writer, buf, schema)
    writer.close()
    return total, n_rows_total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--splits", nargs="+", default=["desktop", "web", "mobile"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2000,
                    help="Parquet write batch size (limits per-worker peak RAM).")
    args = ap.parse_args()

    for split in args.splits:
        in_dir = Path(args.in_root) / split
        out_dir = Path(args.out_root) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(glob.glob(str(in_dir / "train-*.parquet")))
        if not files:
            print(f"  {split}: no train-*.parquet found, skipping")
            continue
        print(f"=== {split}: {len(files)} input files, {args.workers} workers ===")
        total_samples = 0
        total_rows = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_file, f, str(out_dir / os.path.basename(f)),
                            args.seed, split, args.batch_size): f
                for f in files
            }
            for fut in as_completed(futures):
                f = futures[fut]
                n, n_rows = fut.result()
                total_samples += n
                total_rows += n_rows
                print(f"  {os.path.basename(f)}: {n_rows:,} rows → {n:,} samples")
        print(f"  {split} total: {total_rows:,} rows → {total_samples:,} samples")


if __name__ == "__main__":
    main()
