"""Post-hoc compaction, Tier-1 degeneration filter, and fat-shard emission.

Takes the many small ``captions_NNNNN.parquet`` side-table shards and produces:

  1. ``waffle_captions_quarantine.parquet`` — ONE file containing every caption
     the Tier-1 heuristic flagged as degenerate (token loops, number lists,
     confabulated saint lists, etc.) PLUS rows with non-empty ``error``.
     Do not train on this file. Use for audit / regeneration.

  2. ``waffle_captions_clean_compact.parquet`` — ONE thin file (caption side
     table only, no images) for the rows that passed the filter. Small and
     version-trackable independently of the image corpus.

  3. ``waffle_captioned_clean_shards/waffle_captioned_full_NNNNN.parquet`` —
     MULTIPLE ~500 MB fat shards with image_bytes + caption + all WAFFLE
     source metadata joined on ``page_id``. Standard training-dataloader
     layout: parallel I/O across files, per-file random shuffle.

Tier-1 filter (no vision required):
  - ``token_loop``: non-common word repeated >30 times
  - ``quoted_number_list``: 20+ consecutive '"NN,"' fragments
  - ``bulk_number_list``: 40+ consecutive comma-numbers
  - ``tail_repetition``: closing region loops on same token pair
  - plus rows where ``error`` is non-empty (request-level failure)

The "repeated_6gram_3x" detector was removed from the production filter because
it pathologically matches legitimate structural repetition (e.g., a window
schedule that reuses "WINDOW FORM" boilerplate across rows).

Usage:
    python merge_captions.py                       # default: 16 fat shards
    python merge_captions.py --fat-shards 8        # fewer, bigger shards
    python merge_captions.py --no-fat              # skip the fat join entirely
"""

import argparse
import glob
import re
import time
from collections import Counter
from pathlib import Path

import polars as pl


CAPTIONS_DIR_DEFAULT = "/capstor/store/cscs/swissai/infra01/vision-datasets/processed/waffle_captions"
WAFFLE_PERMISSIVE_DIR_DEFAULT = "/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/tau-vailab___WAFFLE/parquet/permissive"


# ------------- Tier-1 degeneration detection ------------------------------

_COMMON = frozenset({
    "the","a","an","and","of","in","is","to","on","with","or","by","at","from",
    "this","that","for","as","are","it","its","each","be","which","these","those",
    "into","onto","within","between","above","below","along","indicating","labeled",
    "column","row","drawing","plan","view","image","sheet","includes","contains",
    "shows","depicts","features","representing","rendered","marked","titled",
    "arranged","positioned","located",
})
_QUOTED_NUM_RE = re.compile(r'(?:"\d+,"\s*){20,}')
_BULK_NUM_RE = re.compile(r'(?:\b\d{1,6}\b,\s*){40,}')
_TAIL_RE = re.compile(r'(\S+\s+\S+)(?:,?\s+\1){5,}')
_WORD_RE = re.compile(r"\b\w+\b")


def degen_reason(caption: str) -> str | None:
    """Return a short flag label if caption is Tier-1 degenerate, else None.

    Reasons are mutually exclusive by priority — the first match wins so we
    don't have to worry about multi-flag rows bloating counts.
    """
    if not caption:
        return "empty"
    # 1. token-loop (highest-precision signal)
    tc = Counter(t.lower() for t in _WORD_RE.findall(caption))
    for tok, n in tc.most_common(5):
        if n > 30 and tok not in _COMMON:
            return f"token_loop:{tok}x{n}"
    # 2. quoted-number list (Hallwyl-style)
    if _QUOTED_NUM_RE.search(caption):
        return "quoted_number_list"
    # 3. bulk comma-number list (Maloof-style)
    if _BULK_NUM_RE.search(caption):
        return "bulk_number_list"
    # 4. tail repetition
    if _TAIL_RE.search(caption[-400:]):
        return "tail_repetition"
    return None


def classify(captions: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split into (clean, quarantine) frames. Quarantine includes both
    error rows (request failed) and Tier-1 flagged rows."""
    reasons = [
        "error:" + err if err else (degen_reason(cap) or "")
        for cap, err in zip(captions["caption"].to_list(),
                            captions["error"].to_list())
    ]
    captions = captions.with_columns(
        pl.Series("quarantine_reason", reasons)
    )
    clean = captions.filter(pl.col("quarantine_reason") == "")
    quarantine = captions.filter(pl.col("quarantine_reason") != "")
    return clean, quarantine


# ------------- Loaders ----------------------------------------------------

def load_captions(captions_dir: Path) -> pl.DataFrame:
    shards = sorted(captions_dir.glob("captions_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no captions_*.parquet under {captions_dir}")
    print(f"reading {len(shards)} caption shards from {captions_dir}", flush=True)
    df = pl.concat([pl.read_parquet(p) for p in shards])
    print(f"  loaded {len(df):,} caption rows", flush=True)
    return df


def load_waffle(waffle_dir: Path) -> pl.DataFrame:
    shards = sorted(waffle_dir.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no train-*.parquet under {waffle_dir}")
    print(f"reading {len(shards)} WAFFLE source shards from {waffle_dir}", flush=True)
    df = pl.concat([pl.read_parquet(p) for p in shards])
    print(f"  loaded {len(df):,} WAFFLE rows", flush=True)
    return df


# ------------- Writers ----------------------------------------------------

def write_single(df: pl.DataFrame, out_path: Path, *, compression: str = "zstd") -> None:
    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression=compression)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  wrote {out_path}  ({len(df):,} rows, {size_mb:.1f} MB, "
          f"{time.time()-t0:.1f}s)", flush=True)


def write_fat_shards(
    clean_caps: pl.DataFrame,
    waffle_dir: Path,
    shard_dir: Path,
    basename: str,
    n_shards: int,
) -> None:
    """Fat training-ready output: image_bytes + caption + source metadata,
    split into N shards for parallel dataloader I/O."""
    waffle = load_waffle(waffle_dir)

    caps_keep = clean_caps.select([
        "page_id", "caption", "word_count",
        pl.col("model").alias("caption_model"),
        pl.col("temperature").alias("caption_temperature"),
        pl.col("max_tokens").alias("caption_max_tokens"),
        pl.col("image_longest_side").alias("caption_image_longest_side"),
    ])

    t0 = time.time()
    joined = waffle.join(caps_keep, on="page_id", how="inner")
    n_dropped = len(waffle) - len(joined)
    print(f"  joined: {len(joined):,} rows; {n_dropped:,} WAFFLE rows had "
          f"no clean caption (either absent or quarantined)", flush=True)

    if n_shards <= 1:
        write_single(joined, shard_dir.parent / f"{basename}.parquet")
        return

    # Shuffle so shard-0 isn't biased by input parquet order (important for
    # random-shuffle training; cheap at this row count).
    joined = joined.sample(fraction=1.0, shuffle=True, seed=0)
    total = len(joined)
    rows_per_shard = (total + n_shards - 1) // n_shards
    print(f"  sharding into {n_shards} fat shards, ~{rows_per_shard:,} rows each",
          flush=True)

    shard_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_shards):
        lo = i * rows_per_shard
        hi = min(lo + rows_per_shard, total)
        if lo >= hi:
            break
        shard = joined.slice(lo, hi - lo)
        out_path = shard_dir / f"{basename}_{i:05d}.parquet"
        shard.write_parquet(out_path, compression="zstd", row_group_size=1000)
        size_mb = out_path.stat().st_size / 1e6
        print(f"    shard {i:2d}: {out_path.name}  ({len(shard):,} rows, "
              f"{size_mb:.1f} MB)", flush=True)
    print(f"  sharding complete in {time.time()-t0:.1f}s", flush=True)


# ------------- Main -------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions-dir", type=Path, default=Path(CAPTIONS_DIR_DEFAULT))
    ap.add_argument("--waffle-dir", type=Path, default=Path(WAFFLE_PERMISSIVE_DIR_DEFAULT))
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Root for output files (default: --captions-dir)")
    ap.add_argument("--fat-shards", type=int, default=16,
                    help="Number of fat (image+caption) shards (default: 16, "
                         "targeting ~300-500 MB per shard for ~11K images). "
                         "Pass 1 for a single monolith.")
    ap.add_argument("--no-fat", action="store_true",
                    help="Skip the fat (image-joined) output; emit only the "
                         "thin compact file and quarantine file.")
    args = ap.parse_args()

    out_dir = args.output_dir or args.captions_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    caps = load_captions(args.captions_dir)
    print("running Tier-1 classifier...", flush=True)
    clean, quarantine = classify(caps)
    print(f"  clean:      {len(clean):,} ({len(clean)/len(caps)*100:.2f}%)", flush=True)
    print(f"  quarantine: {len(quarantine):,} ({len(quarantine)/len(caps)*100:.2f}%)", flush=True)
    print()

    # Breakdown of quarantine reasons for the audit trail
    if len(quarantine):
        reasons = Counter(
            r.split(":", 1)[0] for r in quarantine["quarantine_reason"].to_list()
        )
        print("quarantine reason breakdown:", flush=True)
        for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {k:<24} {v:>5d}  ({v/len(caps)*100:.2f}% of total)", flush=True)
        print()

    # 1. Quarantine: ONE file
    write_single(
        quarantine,
        out_dir / "waffle_captions_quarantine.parquet",
    )

    # 2. Thin clean compact: ONE file (side table, no images)
    write_single(
        clean.drop("quarantine_reason"),
        out_dir / "waffle_captions_clean_compact.parquet",
    )

    # 3. Fat clean: MULTIPLE shards (training-ready)
    if not args.no_fat:
        write_fat_shards(
            clean.drop("quarantine_reason"),
            args.waffle_dir,
            out_dir / "waffle_captioned_clean_shards",
            basename="waffle_captioned_full",
            n_shards=args.fat_shards,
        )


if __name__ == "__main__":
    main()
