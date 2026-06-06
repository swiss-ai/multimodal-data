"""Truncate `art` subset to first turn only.

art has the same-OCR-twice redundancy pattern (28-47% of multi-turn rows have
duplicate user+assistant). Truncating to the first turn eliminates the dup
without losing rows. Single-turn rows are unchanged.

Source: /capstor/.../processed/sft/finevision/art/  (filtered parquets)
Dest:   same dir, in-place rewrite (atomic per-shard via tmp + rename)
"""

from __future__ import annotations
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

ART_DIR = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/art")


def main():
    files = sorted(ART_DIR.glob("train-*.parquet"))
    if not files:
        print(f"no files in {ART_DIR}")
        return 1

    total_in_turns = 0
    total_out_turns = 0
    total_rows = 0
    for f in files:
        tbl = pq.read_table(str(f))
        rows = tbl.to_pydict()
        # Truncate texts to just first turn
        new_texts = []
        for texts in rows["texts"]:
            n_before = len(texts)
            new_texts.append(texts[:1])  # keep only first turn pair
            total_in_turns += n_before
            total_out_turns += 1
            total_rows += 1
        rows["texts"] = new_texts
        new_tbl = pa.Table.from_pydict(rows, schema=tbl.schema)
        tmp = f.with_suffix(".parquet.tmp")
        pq.write_table(new_tbl, str(tmp), compression="zstd")
        tmp.rename(f)
        print(f"  {f.name}: rows={tbl.num_rows:,} (turns {sum(len(t) for t in rows['texts'])})")

    print()
    print(f"=== art: {total_rows:,} rows ===")
    print(f"   turns before: {total_in_turns:,}")
    print(f"   turns after:  {total_out_turns:,} ({total_in_turns - total_out_turns:,} removed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
