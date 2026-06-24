"""Light regex cleanup of FineVision::multihiertt before tokenize.

The processed shards have 5,445 expert-annotated rows from Zhao et al. ACL 2022,
each with a Computations:/Answer: scratchpad. Three cosmetic issues to fix:

1) Every answer starts with a leading '\n' — strip it.
2) Some questions have missing-space typos like "theIntegrated" (3.8% of turns,
   ~208 cases) — split lowercase→Capital boundaries inside words.
3) Trailing period inconsistency on answers — normalize to always end with '.'.

Output goes to processed/sft/finevision/multihiertt_cleaned/ — original is left
intact. Tokenize config points at the _cleaned/ dir.
"""

from __future__ import annotations
import glob
import re
import time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


SRC_DIR = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/multihiertt")
OUT_DIR = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/multihiertt_cleaned")

# Common false positives we should NOT split (proper acronyms, real CamelCase)
PROTECT = {
    "KCP&L", "FactSet", "PowerPoint", "iPhone", "iPad", "eCommerce", "eBay",
    "PayPal", "JPMorgan", "BlackRock", "DataPoint", "macOS", "iOS",
}

TYPO_PATTERN = re.compile(r"\b([a-z]+)([A-Z][a-z]+)\b")


def fix_missing_spaces(s: str) -> str:
    def repl(m):
        full = m.group(0)
        if full in PROTECT:
            return full
        return f"{m.group(1)} {m.group(2)}"
    return TYPO_PATTERN.sub(repl, s)


def normalize_trailing_period(s: str) -> str:
    s = s.rstrip()
    if not s:
        return s
    if not s.endswith("."):
        s = s + "."
    return s


def clean_turn(turn: dict) -> dict:
    u = turn.get("user") or ""
    a = turn.get("assistant") or ""
    return {
        "user": fix_missing_spaces(u.strip()),
        "assistant": normalize_trailing_period(a.lstrip("\n").rstrip()),
    }


def process_shard(src_path: Path, out_path: Path) -> tuple[int, int, int]:
    tbl = pq.read_table(str(src_path))
    rows = tbl.to_pylist()
    n_typo_fixed = 0
    n_lf_stripped = 0
    n_period_added = 0

    for r in rows:
        turns = r.get("texts") or []
        new_turns = []
        for t in turns:
            u_in = t.get("user") or ""
            a_in = t.get("assistant") or ""
            u_out = fix_missing_spaces(u_in.strip())
            a_no_lf = a_in.lstrip("\n").rstrip()
            a_out = normalize_trailing_period(a_no_lf)

            if u_out != u_in:
                n_typo_fixed += 1
            if a_no_lf != a_in:
                n_lf_stripped += 1
            if a_out != a_no_lf:
                n_period_added += 1
            new_turns.append({"user": u_out, "assistant": a_out})
        r["texts"] = new_turns

    new_tbl = pa.Table.from_pylist(rows, schema=tbl.schema)
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(new_tbl, str(tmp), compression="zstd")
    tmp.rename(out_path)
    return n_typo_fixed, n_lf_stripped, n_period_added


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(SRC_DIR.glob("train-*.parquet"))
    print(f"=== cleaning {len(shards)} multihiertt shards ===", flush=True)
    t0 = time.time()
    total_typo = total_lf = total_period = 0
    for sf in shards:
        of = OUT_DIR / sf.name
        nt, nl, np_ = process_shard(sf, of)
        total_typo += nt
        total_lf += nl
        total_period += np_
        print(f"  [{sf.name}] typo_fixes={nt:>4} lf_stripped={nl:>5} period_added={np_:>4}", flush=True)
    print(f"\n=== done in {time.time()-t0:.0f}s ===")
    print(f"  total Q-side typo fixes:      {total_typo:,}")
    print(f"  total A-side leading \\n strips: {total_lf:,}")
    print(f"  total A-side trailing period:  {total_period:,}")


if __name__ == "__main__":
    main()
