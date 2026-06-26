"""Combine Visual Genome annotation JSONs + JPEG bytes into sharded parquet.

Reads:
    annotations/image_data.json          — image_id → URL/dims/local_path
    annotations/region_descriptions.json — 5.4M region phrases with bboxes
    annotations/objects.json             — 3.8M objects with bboxes + names/synsets
    annotations/relationships.json       — 2.3M (subj, predicate, obj) triples
    annotations/attributes.json          — 2.8M object attributes
    images/VG_100K{,_2}/<image_id>.jpg   — 108K JPEGs

Skips (lower-value or huge):
    region_graphs.json (2.6 GB scene-graph variant — redundant w/ regions+objects)
    question_answers.json (separate use case)

Writes:
    raw/stage2/visual_genome/parquet/part-NNNNN-of-NNNNN.parquet

Output schema (one row per image):
    image_id: int64
    image: struct<bytes: binary, path: string>
    width, height: int32
    url: string
    regions:       list<struct<region_id, phrase, x, y, width, height>>
    objects:       list<struct<object_id, names, synsets, x, y, w, h>>
    relationships: list<struct<relationship_id, subject_id, object_id, predicate, synsets>>
    attributes:    list<struct<object_id, attributes, names, x, y, w, h>>
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ──────────────────────────────────────────────────────────────────────────────
# Schema — pinned so partial shards stay compatible
# ──────────────────────────────────────────────────────────────────────────────

REGIONS_T = pa.list_(pa.struct([
    ("region_id", pa.int64()),
    ("phrase",    pa.string()),
    ("x",         pa.int32()),
    ("y",         pa.int32()),
    ("width",     pa.int32()),
    ("height",    pa.int32()),
]))

OBJECTS_T = pa.list_(pa.struct([
    ("object_id", pa.int64()),
    ("names",     pa.list_(pa.string())),
    ("synsets",   pa.list_(pa.string())),
    ("x",         pa.int32()),
    ("y",         pa.int32()),
    ("w",         pa.int32()),
    ("h",         pa.int32()),
]))

RELATIONSHIPS_T = pa.list_(pa.struct([
    ("relationship_id", pa.int64()),
    ("subject_id",      pa.int64()),
    ("object_id",       pa.int64()),
    ("predicate",       pa.string()),
    ("synsets",         pa.list_(pa.string())),
]))

ATTRIBUTES_T = pa.list_(pa.struct([
    ("object_id",  pa.int64()),
    ("attributes", pa.list_(pa.string())),
    ("names",      pa.list_(pa.string())),
    ("x",          pa.int32()),
    ("y",          pa.int32()),
    ("w",          pa.int32()),
    ("h",          pa.int32()),
]))

SCHEMA = pa.schema([
    ("image_id",      pa.int64()),
    ("image",         pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("width",         pa.int32()),
    ("height",        pa.int32()),
    ("url",           pa.string()),
    ("regions",       REGIONS_T),
    ("objects",       OBJECTS_T),
    ("relationships", RELATIONSHIPS_T),
    ("attributes",    ATTRIBUTES_T),
])


def t(start, label):
    print(f"  [{time.time() - start:6.1f}s] {label}", flush=True)


def find_jpeg(image_id: int, image_dirs: list[Path]) -> Path | None:
    for d in image_dirs:
        p = d / f"{image_id}.jpg"
        if p.exists():
            return p
    return None


def normalize_region(r: dict) -> dict:
    return {
        "region_id": int(r.get("region_id") or 0),
        "phrase":    str(r.get("phrase") or ""),
        "x":         int(r.get("x") or 0),
        "y":         int(r.get("y") or 0),
        "width":     int(r.get("width") or 0),
        "height":    int(r.get("height") or 0),
    }


def normalize_object(o: dict) -> dict:
    return {
        "object_id": int(o.get("object_id") or 0),
        "names":     [str(n) for n in (o.get("names") or [])],
        "synsets":   [str(s) for s in (o.get("synsets") or [])],
        "x":         int(o.get("x") or 0),
        "y":         int(o.get("y") or 0),
        "w":         int(o.get("w") or 0),
        "h":         int(o.get("h") or 0),
    }


def normalize_relationship(r: dict) -> dict:
    subj = r.get("subject") or {}
    obj  = r.get("object") or {}
    return {
        "relationship_id": int(r.get("relationship_id") or 0),
        "subject_id":      int(subj.get("object_id") if isinstance(subj, dict) else 0),
        "object_id":       int(obj.get("object_id")  if isinstance(obj, dict)  else 0),
        "predicate":       str(r.get("predicate") or ""),
        "synsets":         [str(s) for s in (r.get("synsets") or [])],
    }


def normalize_attribute_object(o: dict) -> dict:
    return {
        "object_id":  int(o.get("object_id") or 0),
        "attributes": [str(a) for a in (o.get("attributes") or [])],
        "names":      [str(n) for n in (o.get("names") or [])],
        "x":          int(o.get("x") or 0),
        "y":          int(o.get("y") or 0),
        "w":          int(o.get("w") or 0),
        "h":          int(o.get("h") or 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/stage2/visual_genome")
    ap.add_argument("--output", default="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/stage2/visual_genome/parquet")
    ap.add_argument("--shard-rows", type=int, default=2000, help="rows per shard (~250-400 MB at typical VG sizes)")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ann = root / "annotations"
    image_dirs = [root / "images" / "VG_100K", root / "images" / "VG_100K_2"]
    for d in image_dirs:
        if not d.is_dir():
            raise SystemExit(f"missing image dir: {d}")

    start = time.time()
    print("=== loading annotations ===", flush=True)

    t(start, "loading image_data.json")
    image_data = {int(r["image_id"]): r for r in json.load(open(ann / "image_data.json"))}
    print(f"     → {len(image_data):,} images", flush=True)

    t(start, "loading region_descriptions.json")
    region_doc = json.load(open(ann / "region_descriptions.json"))
    regions_by_img = defaultdict(list)
    for d in region_doc:
        img_id = int(d["id"])
        for r in d.get("regions", []):
            regions_by_img[img_id].append(normalize_region(r))
    del region_doc
    print(f"     → regions for {len(regions_by_img):,} images", flush=True)

    t(start, "loading objects.json")
    obj_doc = json.load(open(ann / "objects.json"))
    objects_by_img = defaultdict(list)
    for d in obj_doc:
        img_id = int(d["image_id"])
        for o in d.get("objects", []):
            objects_by_img[img_id].append(normalize_object(o))
    del obj_doc
    print(f"     → objects for {len(objects_by_img):,} images", flush=True)

    t(start, "loading relationships.json")
    rel_doc = json.load(open(ann / "relationships.json"))
    rels_by_img = defaultdict(list)
    for d in rel_doc:
        img_id = int(d["image_id"])
        for r in d.get("relationships", []):
            rels_by_img[img_id].append(normalize_relationship(r))
    del rel_doc
    print(f"     → relationships for {len(rels_by_img):,} images", flush=True)

    t(start, "loading attributes.json")
    attr_doc = json.load(open(ann / "attributes.json"))
    attrs_by_img = defaultdict(list)
    for d in attr_doc:
        img_id = int(d["image_id"])
        for o in d.get("attributes", []):
            attrs_by_img[img_id].append(normalize_attribute_object(o))
    del attr_doc
    print(f"     → attributes for {len(attrs_by_img):,} images", flush=True)

    t(start, "all annotations loaded; writing shards")

    image_ids = sorted(image_data.keys())
    n = len(image_ids)
    n_shards = (n + args.shard_rows - 1) // args.shard_rows
    print(f"\n=== writing {n_shards} shards × ~{args.shard_rows} rows = {n:,} images ===", flush=True)

    n_missing_jpeg = 0
    n_written = 0
    bytes_written = 0
    for shard_idx in range(n_shards):
        lo = shard_idx * args.shard_rows
        hi = min(lo + args.shard_rows, n)
        rows = []
        for img_id in image_ids[lo:hi]:
            meta = image_data[img_id]
            jpeg_path = find_jpeg(img_id, image_dirs)
            if jpeg_path is None:
                n_missing_jpeg += 1
                continue
            try:
                jpeg_bytes = jpeg_path.read_bytes()
            except Exception:
                n_missing_jpeg += 1
                continue
            rows.append({
                "image_id": int(img_id),
                "image": {"bytes": jpeg_bytes, "path": jpeg_path.name},
                "width":  int(meta.get("width") or 0),
                "height": int(meta.get("height") or 0),
                "url":    str(meta.get("url") or ""),
                "regions":       regions_by_img.get(img_id, []),
                "objects":       objects_by_img.get(img_id, []),
                "relationships": rels_by_img.get(img_id, []),
                "attributes":    attrs_by_img.get(img_id, []),
            })
        if not rows:
            continue
        out_path = out_dir / f"part-{shard_idx:05d}-of-{n_shards:05d}.parquet"
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        pq.write_table(table, out_path, compression="zstd")
        sz = out_path.stat().st_size
        bytes_written += sz
        n_written += len(rows)
        if (shard_idx + 1) % 5 == 0 or shard_idx + 1 == n_shards:
            elapsed = time.time() - start
            print(f"  [{elapsed:6.1f}s] shard {shard_idx+1}/{n_shards} — "
                  f"{n_written:,} rows, {bytes_written/1e9:.2f} GB", flush=True)

    total_elapsed = time.time() - start
    print(f"\n=== DONE in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min) ===", flush=True)
    print(f"  rows written:      {n_written:,}")
    print(f"  missing JPEGs:     {n_missing_jpeg}")
    print(f"  total parquet:     {bytes_written/1e9:.2f} GB across {n_shards} shards")
    print(f"  output dir:        {out_dir}")


if __name__ == "__main__":
    main()
