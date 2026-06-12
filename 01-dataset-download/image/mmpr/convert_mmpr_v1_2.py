#!/usr/bin/env python3
"""Convert raw MMPR-v1.2 (OpenGVLab) to the alignment-mode input_parquet schema.

Source (kept compressed, never extracted):
  /capstor/store/cscs/swissai/infra01/vision-datasets/raw/preference/hf___OpenGVLab___MMPR-v1.2/
    annotations.zip           148 jsonl files: {image, question, chosen, rejected, ...}
    images.zip_a{a,b,c}       split zip, 266,908 images (55.6 GB), read via ConcatReader
    meta.json                 annotation -> image-root mapping

Target schema (mllm-dpo.parquet shape, consumed by
vision_tokenization/indexing/alignment/ingest.py ROW_ADAPTERS["preference"]):
  source-id: string
  image:     list<struct<bytes: binary, path: string>>   (marker order)
  prompt:    list<struct<role, content>>  -- single user turn, <image> markers
  accepted:  list<struct<role, content>>  -- single assistant turn (chosen)
  rejected:  list<struct<role, content>>  -- single assistant turn (rejected)

Contract enforcement AT CONVERSION (the in-job scan raises MarkerMismatch and
kills the whole run otherwise — filter here, fail-loud there):
  * per-row markers == n_images (zero-marker rows get one '<image>\\n' per
    image prepended — InternVL loader convention);
  * no '<image>' in chosen/rejected (drop: would crash _parse_preference_row);
  * no pre-existing '<|image|>' anywhere (drop);
  * no empty / identical chosen-rejected pairs (drop: no preference signal);
  * rows referencing images absent from images.zip (drop: 14 RLAIF-V
    Emma_Roberts.jpg refs, removed upstream);
  * self-commentary in CHOSEN ("as an AI I can't view images", ...) dropped by
    default (caption-quality-audit convention); --keep-selfcomm-chosen keeps.
    Self-commentary in REJECTED is intentional preference signal — kept.

Sharding: shard = sha1(first image path) % n_shards, so pairs reusing the same
image colocate (one GPU encode per store, not per shard). Output:
  <out_dir>/part_00.parquet .. part_NN.parquet + _conversion_report.json

Usage (inline on head node; CPU + IO only):
  python scripts/convert_mmpr_v1_2.py                       # full, 8 shards
  python scripts/convert_mmpr_v1_2.py --smoke 300           # smoke.parquet only
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import os
import re
import time
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

RAW = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/raw/preference/"
           "hf___OpenGVLab___MMPR-v1.2")
OUT_DEFAULT = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/"
                   "alignment-processed/mmpr-v1.2")

MARKER = "<image>"
CANONICAL = "<|image|>"
SELFCOMM = re.compile(
    r"(As an AI\b|as an AI language model|I cannot see|I'm unable to view"
    r"|cannot view the image|As a language model"
    r"|I do not have access to the image)", re.I)

MSG = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
SCHEMA = pa.schema([
    pa.field("source-id", pa.string()),
    pa.field("image", pa.list_(pa.struct([("bytes", pa.binary()),
                                          ("path", pa.string())]))),
    pa.field("prompt", MSG),
    pa.field("accepted", MSG),
    pa.field("rejected", MSG),
])

FLUSH_ROWS = 1_000  # ~250 MB/row-group at the 238 KB/ref average


class ConcatReader(io.RawIOBase):
    """Seekable read-only view over split files (images.zip_aa/ab/ac)."""

    def __init__(self, paths):
        self.sizes = [os.path.getsize(p) for p in paths]
        self.offsets = [0]
        for s in self.sizes:
            self.offsets.append(self.offsets[-1] + s)
        self.total = self.offsets[-1]
        self.pos = 0
        self.fhs = [open(p, "rb") for p in paths]

    def seek(self, off, whence=0):
        self.pos = (off if whence == 0 else
                    self.pos + off if whence == 1 else self.total + off)
        return self.pos

    def tell(self):
        return self.pos

    def readable(self):
        return True

    def seekable(self):
        return True

    def read(self, n=-1):
        if n < 0:
            n = self.total - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.total:
            i = bisect.bisect_right(self.offsets, self.pos) - 1
            local = self.pos - self.offsets[i]
            take = min(n, self.sizes[i] - local)
            self.fhs[i].seek(local)
            out += self.fhs[i].read(take)
            self.pos += take
            n -= take
        return bytes(out)


class ShardWriter:
    def __init__(self, path: Path):
        self.path = path
        self.tmp = path.with_suffix(".parquet.tmp")
        self.writer = pq.ParquetWriter(self.tmp, SCHEMA, compression="snappy")
        self.buf: list[dict] = []
        self.rows = 0
        self.bytes = 0

    def add(self, row: dict, n_bytes: int):
        self.buf.append(row)
        self.rows += 1
        self.bytes += n_bytes
        if len(self.buf) >= FLUSH_ROWS:
            self.flush()

    def flush(self):
        if self.buf:
            self.writer.write_table(pa.Table.from_pylist(self.buf, schema=SCHEMA))
            self.buf = []

    def close(self):
        self.flush()
        self.writer.close()
        os.replace(self.tmp, self.path)


def iter_rows(annotations_zip: zipfile.ZipFile, root_of: dict):
    """Yield (source_id, image_zip_paths, question, chosen, rejected)."""
    names = sorted(n for n in annotations_zip.namelist() if n.endswith(".jsonl"))
    for name in names:
        stem = name.rsplit("/", 1)[-1].removesuffix(".jsonl")
        root = root_of[stem + ".jsonl"]
        with annotations_zip.open(name) as f:
            for lineno, line in enumerate(f):
                row = json.loads(line)
                imgs = row["image"] if isinstance(row["image"], list) else [row["image"]]
                paths = [f"{root}/{rel}" for rel in imgs]
                yield (f"mmpr-v1.2-{stem}-{lineno}", paths,
                       row["question"], row["chosen"], row["rejected"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--n-shards", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0,
                    help="write only the first N kept rows to smoke.parquet")
    ap.add_argument("--keep-selfcomm-chosen", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    meta = json.loads((RAW / "meta.json").read_text())
    root_of = {v["annotation"].rsplit("/", 1)[-1]:
               v["root"].removeprefix("MMPR-v1.2/") for v in meta.values()}

    images_zip = zipfile.ZipFile(ConcatReader(
        [RAW / f"images.zip_a{c}" for c in "abc"]))
    have = {i.filename for i in images_zip.infolist() if not i.is_dir()}
    annotations_zip = zipfile.ZipFile(RAW / "annotations.zip")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        writers = [ShardWriter(args.out_dir / "smoke.parquet")]
    else:
        writers = [ShardWriter(args.out_dir / f"part_{i:02d}.parquet")
                   for i in range(args.n_shards)]

    drops = {"missing_image": 0, "marker_in_response": 0, "canonical_marker": 0,
             "empty_response": 0, "identical_pair": 0, "selfcomm_chosen": 0}
    flags = {"zero_marker_prepended": 0, "think_in_chosen": 0,
             "selfcomm_rejected_kept": 0, "multi_image": 0}
    kept = 0
    seen_total = 0

    for source_id, paths, question, chosen, rejected in iter_rows(
            annotations_zip, root_of):
        seen_total += 1
        if any(p not in have for p in paths):
            drops["missing_image"] += 1
            continue
        if CANONICAL in question or CANONICAL in chosen or CANONICAL in rejected:
            drops["canonical_marker"] += 1
            continue
        if MARKER in chosen or MARKER in rejected:
            drops["marker_in_response"] += 1
            continue
        if not chosen.strip() or not rejected.strip():
            drops["empty_response"] += 1
            continue
        if chosen.strip() == rejected.strip():
            drops["identical_pair"] += 1
            continue
        if not args.keep_selfcomm_chosen and SELFCOMM.search(chosen):
            drops["selfcomm_chosen"] += 1
            continue

        n_img = len(paths)
        if question.count(MARKER) == 0:
            question = f"{MARKER}\n" * n_img + question
            flags["zero_marker_prepended"] += 1
        assert question.count(MARKER) == n_img, source_id  # converter invariant

        if "<think>" in chosen:
            flags["think_in_chosen"] += 1
        if SELFCOMM.search(rejected):
            flags["selfcomm_rejected_kept"] += 1
        if n_img > 1:
            flags["multi_image"] += 1

        blobs = []
        n_bytes = 0
        for p in paths:
            with images_zip.open(p) as f:
                b = f.read()
            blobs.append({"bytes": b, "path": p})
            n_bytes += len(b)

        row = {
            "source-id": source_id,
            "image": blobs,
            "prompt": [{"role": "user", "content": question}],
            "accepted": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        }
        if args.smoke:
            writers[0].add(row, n_bytes)
        else:
            shard = int(hashlib.sha1(paths[0].encode()).hexdigest(), 16) % args.n_shards
            writers[shard].add(row, n_bytes)
        kept += 1
        if args.smoke and kept >= args.smoke:
            break
        if kept % 50_000 == 0:
            print(f"[{time.time()-t0:7.0f}s] kept {kept:,} / seen {seen_total:,}",
                  flush=True)

    for w in writers:
        w.close()

    report = {
        "source": str(RAW),
        "rows_seen": seen_total,
        "rows_kept": kept,
        "drops": drops,
        "flags": flags,
        "shards": {w.path.name: {"rows": w.rows, "embedded_bytes": w.bytes}
                   for w in writers},
        "elapsed_s": round(time.time() - t0, 1),
    }
    name = "_smoke_report.json" if args.smoke else "_conversion_report.json"
    (args.out_dir / name).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
