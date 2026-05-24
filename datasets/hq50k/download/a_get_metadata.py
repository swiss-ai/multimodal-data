#!/usr/bin/env python3

import os
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ZIP_PATH = Path(os.environ.get("HQ50K_ZIP_PATH", ""))
OUTPUT_DIR = Path(os.environ.get("METADATA_DIR", "/tmp/metadata/HQ-50K/parquet"))
OUTPUT_PATH = OUTPUT_DIR / "metadata.parquet"
OUTPUT_SCHEMA = pa.schema([("url", pa.string())])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = OUTPUT_DIR / f"{OUTPUT_PATH.name}.tmp"

    if OUTPUT_PATH.exists():
        print(f"skip {OUTPUT_PATH.name}", flush=True)
        return

    if temp_path.exists():
        temp_path.unlink()

    seen = set()
    urls = []

    with zipfile.ZipFile(ZIP_PATH) as z:
        txt_files = [n for n in z.namelist() if n.endswith(".txt")]
        print(f"reading {len(txt_files)} txt files...", flush=True)
        for name in sorted(txt_files):
            with z.open(name) as f:
                for line in f.read().decode().splitlines():
                    url = line.strip()
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    urls.append(url)

    table = pa.table({"url": pa.array(urls, type=pa.string())}, schema=OUTPUT_SCHEMA)
    pq.write_table(table, temp_path)
    temp_path.rename(OUTPUT_PATH)
    print(f"done: {len(urls)} unique URLs -> {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
