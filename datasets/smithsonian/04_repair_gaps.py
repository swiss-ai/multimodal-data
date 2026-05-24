#!/usr/bin/env python3
"""
Repair missing shards caused by the chunk-boundary race condition.
For each gap shard index S in a tier, reads exactly the rows that belong
to shard S from the per-tier parquet and writes them to {S:06d}.tar.
"""

# ── Bootstrap: reuse write_shards from 03_pack_webdataset.py ──────────────────
import importlib.util
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm

spec = importlib.util.spec_from_file_location("packer", Path(__file__).parent / "03_pack_webdataset.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
write_shards = mod.write_shards

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT_ROOT = Path("/path/to/data/vision-datasets/processed/smithsonian")
TIER_DIR = Path("/tmp/toolbox/smithsonian/data/tier_splits")
LOG_DIR = Path("/tmp/toolbox/smithsonian/data")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "repair_gaps.log"),
    ],
)
log = logging.getLogger("repair")

SHARD_SIZE = {"tier1": 500, "tier2": 500, "tier3": 1000, "tier4": 200}


def find_gaps(tier_path: str) -> list[int]:
    d = OUT_ROOT / tier_path
    if not d.exists():
        return []
    tars = sorted(d.glob("*.tar"))
    if not tars:
        return []
    indices = set(int(t.stem) for t in tars)
    max_idx = max(indices)
    return [i for i in range(max_idx + 1) if i not in indices]


def repair_shard(args):
    tier_path, gap_idx, tier_parquet = args
    tier = tier_path.split("/")[0]
    shard_sz = SHARD_SIZE.get(tier, 500)
    row_start = gap_idx * shard_sz
    row_end = row_start + shard_sz

    df = pd.read_parquet(tier_parquet)
    rows = df.iloc[row_start:row_end].to_dict("records")
    result = write_shards(tier_path, rows, shard_start=gap_idx)
    return (tier_path, gap_idx, result["written"], result["skipped"])


def main():
    nmnh_tiers = [
        "tier3/nmnh/anthro",
        "tier3/nmnh/birds",
        "tier3/nmnh/botany",
        "tier3/nmnh/ento",
        "tier3/nmnh/fishes",
        "tier3/nmnh/herps",
        "tier3/nmnh/inv",
        "tier3/nmnh/mammals",
        "tier3/nmnh/minsci",
        "tier3/nmnh/paleo",
    ]

    tasks = []
    for tp in nmnh_tiers:
        gaps = find_gaps(tp)
        if not gaps:
            log.info(f"  {tp}: no gaps")
            continue
        parquet = TIER_DIR / f"{tp.replace('/', '__')}.parquet"
        log.info(f"  {tp}: {len(gaps)} gaps → {gaps[:5]}{'...' if len(gaps) > 5 else ''}")
        for g in gaps:
            tasks.append((tp, g, parquet))

    if not tasks:
        log.info("No gaps found — nothing to repair.")
        return

    log.info(f"\nRepairing {len(tasks)} missing shards with {min(len(tasks), 64)} workers…")
    n_workers = min(len(tasks), 64)

    with Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(repair_shard, tasks),
                total=len(tasks),
                desc="Repairing shards",
            )
        )

    total_written = sum(r[2] for r in results)
    total_skipped = sum(r[3] for r in results)
    log.info(f"\nRepair complete: {total_written:,} written, {total_skipped:,} skipped across {len(results)} shards")

    # Final gap check
    log.info("\nPost-repair gap check:")
    all_ok = True
    for tp in nmnh_tiers:
        gaps = find_gaps(tp)
        status = "OK" if not gaps else f"FAIL: {len(gaps)} gaps remain: {gaps}"
        log.info(f"  {tp}: {status}")
        if gaps:
            all_ok = False
    log.info("\nOVERALL: " + ("ALL GAPS FILLED" if all_ok else "STILL HAS GAPS"))


if __name__ == "__main__":
    main()
