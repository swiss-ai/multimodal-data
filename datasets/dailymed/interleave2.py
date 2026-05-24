"""
Build interleaved parquet shards with three deduplication passes.

  Dedup 1 (within-doc, text):   skip repeated text segments (exact hash)
  Dedup 2 (within-doc, images): skip consecutive image segments whose captions
                                  share word Jaccard >= CAPTION_SIM_THRESHOLD
  Dedup 3 (cross-doc):          MinHash LSH to drop near-duplicate documents
                                  (estimated Jaccard >= MINHASH_THRESHOLD)

Output schema (one row per document):
    id, n_segments, n_images, n_text_chars, segments_json, images_bytes

Run:
    .venv/bin/python interleave2.py
"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

CAPTION_DIR = "/path/to/data/medical-datasets/raw/dailymed_spl/parquet_caption"
SRC_DIR = "/path/to/data/medical-datasets/raw/dailymed_spl/parquet_md"
DST_DIR = "/path/to/data/medical-datasets/raw/dailymed_spl/interleaved2"

MIN_TEXT_CHARS = 2000
CAPTION_SIM_THRESH = 0.40  # dedup 2
MINHASH_THRESH = 0.80  # dedup 3
N_PERM = 128
LSH_BANDS = 16
LSH_ROWS = 8  # N_PERM == LSH_BANDS * LSH_ROWS
SHINGLE_K = 5

IMAGE_RE = re.compile(r"\[\[IMAGE: [^\]|]+ \| ([^\]]+)\]\]")

SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("n_segments", pa.int32()),
        pa.field("n_images", pa.int32()),
        pa.field("n_text_chars", pa.int32()),
        pa.field("segments_json", pa.large_string()),
        pa.field("images_bytes", pa.list_(pa.large_binary())),
    ]
)

# Fixed random coefficients for MinHash (reproducible across runs)
_RNG = np.random.default_rng(42)
_A = _RNG.integers(1, 2**32, size=N_PERM, dtype=np.uint64)
_B = _RNG.integers(0, 2**32, size=N_PERM, dtype=np.uint64)


# ── MinHash / LSH ─────────────────────────────────────────────────────────────


def compute_minhash(text: str) -> np.ndarray | None:
    words = re.sub(r"\W+", " ", text.lower()).split()
    shingles = list({"_".join(words[i : i + SHINGLE_K]) for i in range(len(words) - SHINGLE_K + 1)})
    if len(shingles) < 10:
        return None
    base = np.array(
        [int(hashlib.md5(s.encode()).hexdigest()[:16], 16) & 0xFFFFFFFFFFFFFFFF for s in shingles],
        dtype=np.uint64,
    )
    # (N_PERM, n_shingles) – uint64 wraps naturally, giving N_PERM independent hash fns
    hashes = _A[:, None] * base[None, :] + _B[:, None]
    return hashes.min(axis=1)


def jaccard_from_sig(s1: np.ndarray, s2: np.ndarray) -> float:
    return float(np.mean(s1 == s2))


def find_cross_doc_duplicates(signatures: dict) -> set:
    """LSH to find near-duplicate doc_ids; returns the set to DROP."""
    buckets: dict = defaultdict(list)
    for doc_id, sig in signatures.items():
        if sig is None:
            continue
        for b in range(LSH_BANDS):
            key = (b, sig[b * LSH_ROWS : (b + 1) * LSH_ROWS].tobytes())
            buckets[key].append(doc_id)

    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    n_pairs = 0
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, min(len(bucket), i + 50)):  # cap to avoid O(n²) on large buckets
                si, sj = signatures[bucket[i]], signatures[bucket[j]]
                if si is not None and sj is not None:
                    if jaccard_from_sig(si, sj) >= MINHASH_THRESH:
                        union(bucket[i], bucket[j])
                        n_pairs += 1

    clusters: dict = defaultdict(list)
    for doc_id in signatures:
        clusters[find(doc_id)].append(doc_id)

    to_drop: set = set()
    n_dup_clusters = 0
    for cluster in clusters.values():
        if len(cluster) > 1:
            n_dup_clusters += 1
            cluster.sort()  # keep lexicographically first (earliest date in ID)
            to_drop.update(cluster[1:])

    print(f"[dedup3] {n_pairs} duplicate pairs → {n_dup_clusters} clusters → dropping {len(to_drop)} docs")
    return to_drop


# ── Caption similarity (dedup 2) ──────────────────────────────────────────────


def word_jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ── Segment builder (dedup 1 + 2) ────────────────────────────────────────────


def parse_segments(markdown: str, doc_id: str, images_by_name: dict, captions: dict):
    parts = IMAGE_RE.split(markdown)
    seen_hashes: set = set()
    segments = []
    images_bytes_out = []
    img_idx = 0
    last_alt = ""

    i = 0
    while i < len(parts):
        text = parts[i].strip()
        if len(text) >= MIN_TEXT_CHARS:
            h = hashlib.md5(text.encode()).hexdigest()
            if h not in seen_hashes:  # dedup 1
                seen_hashes.add(h)
                segments.append({"type": "text", "value": text})
            last_alt = ""  # reset: images separated by text are not consecutive
        i += 1

        if i < len(parts):
            filename = parts[i].strip()
            alt = captions.get((doc_id, filename), "")
            raw = images_by_name.get(filename, b"")
            if not isinstance(raw, bytes):
                raw = bytes(raw)

            # dedup 2: skip truly consecutive images with nearly identical captions
            if alt and last_alt and word_jaccard(alt, last_alt) >= CAPTION_SIM_THRESH:
                i += 1
                continue

            segments.append(
                {
                    "type": "image",
                    "image_index": img_idx,
                    "filename": filename,
                    "alt": alt,
                }
            )
            images_bytes_out.append(raw)
            img_idx += 1
            last_alt = alt
            i += 1

    return segments, images_bytes_out


# ── Caption loading ───────────────────────────────────────────────────────────


def load_captions() -> dict:
    caps = {}
    files = sorted(Path(CAPTION_DIR).glob("task_*.jsonl"))
    print(f"[interleave2] loading captions from {len(files)} files...")
    for jf in tqdm(files, unit="file"):
        with open(jf, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                caps[(r["doc_id"], r["image_name"])] = r["caption"]
    print(f"[interleave2] {len(caps)} captions loaded")
    return caps


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    Path(DST_DIR).mkdir(parents=True, exist_ok=True)
    shards = sorted(Path(SRC_DIR).glob("part-*.parquet"))

    # Phase 1: MinHash signatures (markdown only — fast pass)
    print(f"[dedup3] computing signatures for {len(shards)} shards...")
    signatures: dict = {}
    for shard in tqdm(shards, unit="shard"):
        t = pq.read_table(str(shard), columns=["id", "markdown"])
        for row in t.to_pylist():
            signatures[row["id"]] = compute_minhash(row["markdown"] or "")
    print(f"[dedup3] {len(signatures)} documents fingerprinted")

    # Phase 2: cross-doc duplicate detection
    to_drop = find_cross_doc_duplicates(signatures)

    # Phase 3: build interleaved output
    captions = load_captions()
    print(f"[interleave2] building {len(shards)} shards → {DST_DIR}")

    total_docs = total_imgs = total_dropped = 0
    for shard in tqdm(shards, unit="shard"):
        out_path = Path(DST_DIR) / shard.name
        if out_path.exists():
            print(f"[interleave2] {shard.name} already exists, skipping")
            continue

        t = pq.read_table(str(shard), columns=["id", "markdown", "images"])
        out_rows = []
        for row in t.to_pylist():
            doc_id = row["id"]
            if doc_id in to_drop:
                total_dropped += 1
                continue
            markdown = row.get("markdown") or ""
            images_by_name = {img["name"]: img["bytes"] for img in (row.get("images") or [])}
            segments, img_bytes = parse_segments(markdown, doc_id, images_by_name, captions)
            n_images = sum(1 for s in segments if s["type"] == "image")
            n_text_chars = sum(len(s["value"]) for s in segments if s["type"] == "text")
            out_rows.append(
                {
                    "id": doc_id,
                    "n_segments": len(segments),
                    "n_images": n_images,
                    "n_text_chars": n_text_chars,
                    "segments_json": json.dumps(segments, ensure_ascii=False),
                    "images_bytes": img_bytes,
                }
            )
            total_docs += 1
            total_imgs += n_images

        table = pa.Table.from_pylist(out_rows, schema=SCHEMA)
        pq.write_table(table, str(out_path), compression="zstd", compression_level=3)
        print(
            f"[interleave2] {out_path.name}: {len(out_rows)} docs, "
            f"{sum(r['n_images'] for r in out_rows)} images, "
            f"{out_path.stat().st_size / 1e6:.0f} MB"
        )

    print(
        f"[interleave2] done — {total_docs} docs, {total_imgs} images kept, "
        f"{total_dropped} docs dropped by cross-doc dedup"
    )


if __name__ == "__main__":
    main()
