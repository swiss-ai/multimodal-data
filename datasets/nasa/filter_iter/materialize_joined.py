"""Materialize the joined (captions + images) dataset as sharded parquet.

Parallelism: bounded multiprocessing.Pool(N) over source shards. Each worker:
  - scans one source parquet shard
  - filters to target nasa_ids
  - writes its own part-<src_shard_idx>.parquet (no write contention)

Main process: orchestrates + aggregates progress. Workers are resume-safe:
if a worker's output already exists, the shard is skipped.

Schema per row: nasa_id | caption | verdict | seed | title | orig_description |
                license | center | date_created | image_width | image_height |
                image_url | image_bytes
"""
import glob
import os
import sys
import time
from multiprocessing import Pool, set_start_method
from pathlib import Path

import polars as pl

CAPTIONS = "/capstor/scratch/cscs/xyixuan/nasa_production_partial/production_captions.parquet"
SHARDS_GLOB = "/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/web___nasa___images/shards/*.parquet"
OUT_DIR = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/nasa_cooldown/qwen3p6_27b_image_captioned")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_WORKERS = 12  # dropped from 36 after OOM; 12 × ~15 GB peak = ~180 GB, safe
SELECT_COLS = [
    "nasa_id", "title", "description", "center", "date_created",
    "image_url", "image_width", "image_height", "image_bytes", "license",
]


def _load_caps_lookup():
    """Load captions dict once in main process; passed to workers as initializer."""
    df = pl.read_parquet(CAPTIONS).filter(pl.col("verdict").is_not_null())
    return {r["nasa_id"]: (r["caption"], r["verdict"], r["seed"])
            for r in df.iter_rows(named=True)}


_CAPS = None
def _init_worker(caps_dict):
    global _CAPS
    _CAPS = caps_dict


def _process_shard(shard_path):
    """Worker: scan one source shard, join captions, write own output parquet.
    Returns (shard_name, rows_written, out_size_bytes, error_or_None)."""
    shard_name = Path(shard_path).name
    shard_idx = shard_name.split("_")[-1].split(".")[0]  # e.g. '00042'
    out_path = OUT_DIR / f"part-{shard_idx}.parquet"
    # Resume: skip if output already exists
    if out_path.exists() and out_path.stat().st_size > 0:
        return (shard_name, -1, out_path.stat().st_size, "already_exists")

    try:
        df = (pl.scan_parquet(shard_path)
                .filter(pl.col("nasa_id").is_in(list(_CAPS.keys())))
                .select(SELECT_COLS)
                .collect(engine="streaming"))
    except Exception as e:
        return (shard_name, 0, 0, f"scan_failed: {type(e).__name__}: {e}")

    if len(df) == 0:
        # Write empty marker so resume skips it
        return (shard_name, 0, 0, "no_matches")

    enriched = []
    for row in df.iter_rows(named=True):
        nid = row["nasa_id"]
        cap = _CAPS.get(nid)
        if cap is None:
            continue
        caption, verdict, seed = cap
        enriched.append({
            "nasa_id":          nid,
            "caption":          caption,
            "verdict":          verdict,
            "seed":             seed,
            "title":            row["title"],
            "orig_description": row["description"],
            "license":          row["license"],
            "center":           row["center"],
            "date_created":     row["date_created"],
            "image_width":      row["image_width"],
            "image_height":     row["image_height"],
            "image_url":        row["image_url"],
            "image_bytes":      row["image_bytes"],
        })

    if not enriched:
        return (shard_name, 0, 0, "no_matches_post_join")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        # Write via pyarrow directly (not polars) with small row groups so
        # thrift page headers stay well under pyarrow's read-side size limit.
        # Each row group holds ~50 rows × ~30 MB image ≈ 1.5 GB — comfortably
        # readable by pyarrow's default thrift_string_size_limit of 100 MB.
        out_df = pl.DataFrame(enriched, strict=False)
        table = out_df.to_arrow()
        tmp = out_path.with_suffix(".tmp")
        pq.write_table(
            table, tmp,
            compression="zstd",
            compression_level=3,
            row_group_size=50,   # bound row groups by row count
            use_dictionary=False,
        )
        os.replace(tmp, out_path)
    except Exception as e:
        return (shard_name, 0, 0, f"write_failed: {type(e).__name__}: {e}")

    # Pyarrow round-trip smoke test — fail loud per-shard instead of shipping
    # a pyarrow-unreadable dataset that downstream loaders choke on.
    try:
        pf = pq.ParquetFile(out_path)
        _ = pf.read_row_group(0, columns=['image_bytes', 'caption'])
    except Exception as e:
        out_path.unlink(missing_ok=True)
        return (shard_name, 0, 0, f"pyarrow_roundtrip_failed: {type(e).__name__}: {e}")

    return (shard_name, len(enriched), out_path.stat().st_size, None)


def main():
    # 'spawn' avoids the polars + fork deadlock: polars' Rayon thread pool in
    # the main process gets copied into children with dead threads holding
    # locks, causing every worker to block on futex_wait forever. Spawn creates
    # fresh Python processes with a clean polars thread pool.
    set_start_method("spawn", force=True)

    print(f"loading captions from {CAPTIONS}")
    caps = _load_caps_lookup()
    print(f"  {len(caps)} captions (error rows already filtered)")

    shards = sorted(glob.glob(SHARDS_GLOB))
    print(f"streaming over {len(shards)} source shards with Pool({N_WORKERS})")

    t0 = time.time()
    total_rows = 0
    total_bytes = 0
    n_skipped = 0
    n_err = 0
    processed = 0

    with Pool(N_WORKERS, initializer=_init_worker, initargs=(caps,)) as pool:
        for shard_name, rows_written, size_b, err in pool.imap_unordered(_process_shard, shards, chunksize=1):
            processed += 1
            if err == "already_exists":
                n_skipped += 1
            elif err:
                n_err += 1
                print(f"  [err] {shard_name}: {err}", flush=True)
            else:
                total_rows += rows_written
                total_bytes += size_b
            if processed % 20 == 0 or processed == len(shards):
                elapsed = time.time() - t0
                rate_shards = processed / max(elapsed, 0.1)
                eta_s = (len(shards) - processed) / max(rate_shards, 0.001)
                print(f"  [{processed}/{len(shards)}] elapsed={elapsed:.0f}s  "
                      f"rows_written={total_rows}  size_gb={total_bytes/1024**3:.1f}  "
                      f"skipped={n_skipped}  errs={n_err}  eta={eta_s/60:.1f}m", flush=True)

    print(f"\n✅ done. total rows: {total_rows}, total size: {total_bytes/1024**3:.2f} GB, "
          f"wall: {(time.time()-t0)/60:.1f} min")
    print(f"output: {OUT_DIR}")


if __name__ == "__main__":
    main()
