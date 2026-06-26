"""Decontaminate ai2d_merged against the standard AI2D evaluation test set.

Downloads `lmms-lab/ai2d` (3,088 rows used by lmms-eval / MMMU-style benchmarks)
and SHA256-hashes the image bytes. Then hashes every image in our ai2d_merged
parquets and drops any row whose hash matches a test image.

Output: processed/sft/finevision/ai2d_merged_decontaminated/

Note: SHA256 catches exact-byte matches. If FineVision's ai2d_merged contains
crops/augmentations of AI2D-test images (not byte-identical), those won't be
caught by SHA256 — would need perceptual hash (e.g., dHash) or SSCD embeddings
for that level. SHA256 is the safe floor.
"""

from __future__ import annotations
import hashlib
import io
import os
import sys
import time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# 1. Get lmms-lab/ai2d test set (small, ~140 MB)
LMMS_AI2D_DIR = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/hf_downloads/lmms-lab_ai2d")
SRC_DIR = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/hf_downloads/finevision/ai2d_merged")
OUT_DIR = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/ai2d_merged_decontaminated")


def download_test_set():
    if not LMMS_AI2D_DIR.exists() or not list(LMMS_AI2D_DIR.glob("**/*.parquet")):
        print(f"downloading lmms-lab/ai2d test set...", flush=True)
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        os.system(
            f"/capstor/scratch/cscs/xyixuan/venvs/hftools/bin/hf download lmms-lab/ai2d "
            f"--repo-type=dataset --local-dir {LMMS_AI2D_DIR} 2>&1 | tail -3"
        )
    files = sorted(LMMS_AI2D_DIR.glob("**/*.parquet"))
    print(f"  test parquet files: {len(files)}", flush=True)
    return files


def hash_image_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_test_hash_set(test_files):
    test_hashes = set()
    for f in test_files:
        tbl = pq.read_table(str(f), columns=["image"])
        for img in tbl["image"].to_pylist():
            # img is dict with bytes key
            b = img["bytes"] if isinstance(img, dict) else img
            test_hashes.add(hash_image_bytes(b))
    print(f"  hashed {len(test_hashes):,} test images", flush=True)
    return test_hashes


def decontaminate_shard(shard_path: Path, out_path: Path, test_hashes: set) -> dict:
    tbl = pq.read_table(str(shard_path))
    rows = tbl.to_pylist()
    n_in = len(rows)
    keep = []
    n_contaminated = 0
    for row in rows:
        # ai2d_merged: images is list of struct{bytes, path}
        try:
            b = row["images"][0]["bytes"]
            if hash_image_bytes(b) in test_hashes:
                n_contaminated += 1
                continue
            keep.append(row)
        except Exception:
            continue
    if keep:
        new_tbl = pa.Table.from_pylist(keep, schema=tbl.schema)
        tmp = out_path.with_suffix(".parquet.tmp")
        pq.write_table(new_tbl, str(tmp), compression="zstd")
        tmp.rename(out_path)
    return {"in": n_in, "out": len(keep), "contaminated": n_contaminated}


def main():
    t0 = time.time()
    print("=== decontaminate ai2d_merged against lmms-lab/ai2d ===", flush=True)
    test_files = download_test_set()
    test_hashes = build_test_hash_set(test_files)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(SRC_DIR.glob("train-*.parquet"))
    print(f"\nfiltering {len(shards)} ai2d_merged shards", flush=True)
    total_in = total_out = total_cont = 0
    for sf in shards:
        out_path = OUT_DIR / sf.name
        stats = decontaminate_shard(sf, out_path, test_hashes)
        total_in += stats["in"]
        total_out += stats["out"]
        total_cont += stats["contaminated"]
        pct = 100 * stats["contaminated"] / max(1, stats["in"])
        print(f"  [{sf.name}] in={stats['in']:>5,} out={stats['out']:>5,} "
              f"contaminated={stats['contaminated']:>3,} ({pct:.1f}%)", flush=True)

    print(f"\n=== done in {time.time()-t0:.0f}s ===")
    print(f"  in:           {total_in:,}")
    print(f"  out:          {total_out:,}")
    print(f"  contaminated: {total_cont:,} ({100*total_cont/max(1,total_in):.1f}%)")


if __name__ == "__main__":
    main()
