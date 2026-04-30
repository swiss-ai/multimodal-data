"""Fold regenerated / manually-audited captions back into the clean training set.

Inputs (all optional — script folds whichever are present):
  - ``waffle_captions_clean_compact.parquet``  (baseline from merge_captions.py)
  - ``waffle_captions_regenerated.parquet``    (temperature=0 retry; only
                                                 rows with quarantine_reason == "" are folded)
  - ``captions.jsonl``                          (Claude-audited manual captions,
                                                 page_ids that remained degenerate)

Semantics:
  - The baseline clean compact is the starting set.
  - Regenerated clean rows UPSERT into the clean set by page_id.
    (If the original baseline already has that page_id — which it wouldn't,
    since retries target quarantined page_ids — the regenerated row wins.)
  - JSONL rows UPSERT likewise, and win over regenerated rows for the same
    page_id (manual audit > model retry > baseline).
  - Output:
      * ``waffle_captions_clean_compact.parquet`` (rewritten in place with union)
      * ``waffle_captioned_clean_shards/waffle_captioned_full_NNNNN.parquet``
        (re-sharded training files)
      * ``waffle_captions_still_quarantined.parquet`` (unchanged; kept for audit)

Usage:
    python merge_regenerated.py
    python merge_regenerated.py --fat-shards 16
"""

import argparse
import json
from pathlib import Path

import polars as pl

from merge_captions import (
    CAPTIONS_DIR_DEFAULT,
    WAFFLE_PERMISSIVE_DIR_DEFAULT,
    write_fat_shards,
    write_single,
)


def load_jsonl_captions(path: Path) -> pl.DataFrame | None:
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not rows:
        return None
    df = pl.DataFrame(rows)
    # Match the schema used elsewhere — drop the quarantine_reason concept
    # entirely since these rows are explicitly trusted.
    expected_cols = [
        "page_id", "bucket", "caption", "latency_s", "word_count",
        "license_id", "model", "temperature", "max_tokens",
        "image_longest_side", "error",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"JSONL {path} is missing columns: {missing}. Expected schema: {expected_cols}"
        )
    return df.select(expected_cols)


def upsert(base: pl.DataFrame, new: pl.DataFrame) -> pl.DataFrame:
    """Return base ∪ new, with new winning for duplicate page_ids."""
    # Align schemas to base's column order; if new has extra cols (e.g.
    # quarantine_reason), drop them. If new is missing cols, fill empty.
    cols = base.columns
    new2 = new.select([c for c in cols if c in new.columns])
    for c in cols:
        if c not in new2.columns:
            # Use a sensible null; polars concat_diag can handle this but
            # explicit is clearer.
            new2 = new2.with_columns(pl.lit(None).alias(c))
    new2 = new2.select(cols)
    base_minus = base.filter(~pl.col("page_id").is_in(new["page_id"]))
    return pl.concat([base_minus, new2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions-dir", type=Path, default=Path(CAPTIONS_DIR_DEFAULT))
    ap.add_argument("--waffle-dir", type=Path, default=Path(WAFFLE_PERMISSIVE_DIR_DEFAULT))
    ap.add_argument("--jsonl", type=Path,
                    default=Path("/capstor/scratch/cscs/xyixuan/waffle_claude_captions/captions.jsonl"),
                    help="Claude-audited manual captions (optional).")
    ap.add_argument("--fat-shards", type=int, default=16)
    args = ap.parse_args()

    clean_path = args.captions_dir / "waffle_captions_clean_compact.parquet"
    regen_path = args.captions_dir / "waffle_captions_regenerated.parquet"

    if not clean_path.is_file():
        raise FileNotFoundError(f"baseline {clean_path} is missing — run merge_captions.py first")

    base = pl.read_parquet(clean_path)
    # Defensive dedupe — the first-pass merge occasionally double-wrote a
    # row (identical content, duplicate page_id). Keep the first occurrence.
    before = len(base)
    base = base.unique(subset=["page_id"], keep="first", maintain_order=True)
    if len(base) != before:
        print(f"  deduped baseline: {before:,} -> {len(base):,} rows "
              f"(removed {before - len(base)} duplicate page_ids)", flush=True)
    print(f"baseline clean: {len(base):,} rows", flush=True)

    merged = base
    total_added = 0

    # Regenerated: keep only those that passed the retry Tier-1 filter.
    if regen_path.is_file():
        regen = pl.read_parquet(regen_path)
        regen_clean = regen.filter(pl.col("quarantine_reason") == "")
        print(f"regenerated clean: {len(regen_clean):,} rows "
              f"(of {len(regen)} retries)", flush=True)
        before = len(merged)
        merged = upsert(merged, regen_clean)
        total_added += len(merged) - before
    else:
        print(f"no regenerated parquet at {regen_path}; skipping", flush=True)

    # Claude-audited JSONL.
    jsonl_df = load_jsonl_captions(args.jsonl)
    if jsonl_df is not None:
        print(f"claude-audited JSONL: {len(jsonl_df):,} rows", flush=True)
        before = len(merged)
        merged = upsert(merged, jsonl_df)
        total_added += len(merged) - before
    else:
        print(f"no JSONL audit at {args.jsonl}; skipping", flush=True)

    print(f"\nfinal clean set: {len(merged):,} rows "
          f"(+{total_added} over baseline)", flush=True)

    # Rewrite the clean compact (in place)
    write_single(merged, clean_path)

    # Re-shard the fat training output
    write_fat_shards(
        merged,
        args.waffle_dir,
        args.captions_dir / "waffle_captioned_clean_shards",
        basename="waffle_captioned_full",
        n_shards=args.fat_shards,
    )


if __name__ == "__main__":
    main()
