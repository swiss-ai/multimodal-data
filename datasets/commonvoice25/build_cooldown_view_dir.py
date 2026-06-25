#!/usr/bin/env python3
"""Symlink-only view directory for the CommonVoice-25 cooldown subset.

The cooldown pipeline wants only the `train.parquet` + `validated_extra.parquet`
splits per language; the canonical processed dir at
`/processed/commonvoice25/<lang>/` also contains `dev/test/other/invalidated`
parquets. The convert pipeline accepts a single glob pattern (no brace
expansion), so a clean way to scope the input is a separate view dir whose
*-only contents are the desired files via symlinks.

Output:
    /processed/commonvoice25_cooldown/<lang>/train.parquet
    /processed/commonvoice25_cooldown/<lang>/validated_extra.parquet

Then `cooldown_commonvoice.yaml` points at this dir with files: */*.parquet.
"""

from __future__ import annotations

from pathlib import Path

SRC_ROOT = Path("/capstor/store/cscs/swissai/infra01/audio-datasets/processed/commonvoice25")
DST_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/audio-datasets/processed/commonvoice25_cooldown"
)
WANTED = ("train.parquet", "validated_extra.parquet")


def main() -> None:
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    langs = sorted(p.name for p in SRC_ROOT.iterdir() if p.is_dir())
    n_links = 0
    n_missing = 0
    for lang in langs:
        src_dir = SRC_ROOT / lang
        dst_dir = DST_ROOT / lang
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fname in WANTED:
            src = src_dir / fname
            dst = dst_dir / fname
            if not src.is_file():
                print(f"  MISSING: {src}")
                n_missing += 1
                continue
            if dst.is_symlink() or dst.exists():
                # idempotent: skip if already linked
                continue
            dst.symlink_to(src)
            n_links += 1
    print(f"\nlangs: {len(langs)}")
    print(f"new symlinks: {n_links}")
    print(f"missing: {n_missing}")
    print(f"view dir: {DST_ROOT}")


if __name__ == "__main__":
    main()
