#!/usr/bin/env python3

import csv
from pathlib import Path

import duckdb
import pyarrow as pa

DATASET_DIR = Path("/path/to/data/vision-datasets/hf___madebyollin___megalith-10m/data")
CAPTION_CSV = Path("/path/to/data/vision-datasets/hf___aipicasso___megalith-10m-florence2/metadata.csv")
ROOT_DIR = Path("/tmp/metadata/megalith_10m_florence2")
WORK_DIR = ROOT_DIR / "working"
OUTPUT_DIR = ROOT_DIR / "parquet"
DB_PATH = WORK_DIR / "metadata.duckdb"
CSV_BATCH_SIZE = 100_000
THREADS = 32
STATE_KEY = "captions_loaded"


def sql_quote(value):
    return str(value).replace("'", "''")


def iter_caption_batches():
    with CAPTION_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        url_sources = []
        captions = []

        for row in reader:
            url_source = (row.get("url_source") or "").strip()
            if not url_source:
                continue

            url_sources.append(url_source)
            captions.append((row.get("caption") or "").strip())

            if len(url_sources) == CSV_BATCH_SIZE:
                yield pa.table(
                    {"url_source": url_sources, "caption": captions},
                    schema=pa.schema([("url_source", pa.string()), ("caption", pa.string())]),
                )
                url_sources = []
                captions = []

        if url_sources:
            yield pa.table(
                {"url_source": url_sources, "caption": captions},
                schema=pa.schema([("url_source", pa.string()), ("caption", pa.string())]),
            )


def connect():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"PRAGMA threads={THREADS}")
    return con


def captions_ready(con):
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_state (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
        )
        """
    )
    row = con.execute("SELECT value FROM pipeline_state WHERE key = ?", [STATE_KEY]).fetchone()
    return row is not None


def load_captions(con):
    if captions_ready(con):
        row_count = con.execute("SELECT COUNT(*) FROM captions").fetchone()[0]
        print(f"captions table already loaded rows={row_count}", flush=True)
        return

    con.execute("DROP TABLE IF EXISTS captions_raw")
    con.execute("DROP TABLE IF EXISTS captions")
    con.execute("CREATE TABLE captions_raw (url_source VARCHAR, caption VARCHAR)")
    con.execute("BEGIN TRANSACTION")

    rows_loaded = 0
    for batch_index, batch in enumerate(iter_caption_batches(), start=1):
        view_name = f"caption_batch_{batch_index}"
        con.register(view_name, batch)
        con.execute(f"INSERT INTO captions_raw SELECT * FROM {view_name}")
        con.unregister(view_name)
        rows_loaded += batch.num_rows
        if batch_index % 10 == 0:
            print(f"loaded caption rows={rows_loaded}", flush=True)

    con.execute("COMMIT")

    con.execute(
        """
        CREATE TABLE captions AS
        SELECT url_source, any_value(caption) AS caption
        FROM captions_raw
        GROUP BY url_source
        """
    )
    con.execute("DROP TABLE captions_raw")
    con.execute(
        "INSERT OR REPLACE INTO pipeline_state (key, value) VALUES (?, ?)",
        [STATE_KEY, str(rows_loaded)],
    )

    deduped_rows = con.execute("SELECT COUNT(*) FROM captions").fetchone()[0]
    print(
        f"captions loaded raw_rows={rows_loaded} unique_url_sources={deduped_rows}",
        flush=True,
    )


def output_path_for(parquet_path):
    part_id = parquet_path.stem.split("-")[1]
    return OUTPUT_DIR / f"metadata_{part_id}.parquet"


def export_metadata(con, parquet_path):
    output_path = output_path_for(parquet_path)
    temp_path = OUTPUT_DIR / f"{output_path.name}.tmp"
    if output_path.exists():
        print(f"skip {output_path.name}", flush=True)
        return

    if temp_path.exists():
        temp_path.unlink()

    source_path = sql_quote(parquet_path)
    temp_output = sql_quote(temp_path)
    row_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{source_path}') AS m
        JOIN captions AS c
          ON m.url_source = c.url_source
        """
    ).fetchone()[0]

    # Prefer the high-resolution image when Megalith provides one.
    con.execute(
        f"""
        COPY (
            SELECT
                COALESCE(NULLIF(m.url_highres, ''), m.url) AS url_highres,
                m.url_source AS url_source,
                c.caption AS caption
            FROM read_parquet('{source_path}') AS m
            JOIN captions AS c
              ON m.url_source = c.url_source
        )
        TO '{temp_output}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    temp_path.rename(output_path)
    print(f"{output_path.name} rows={row_count}", flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(DATASET_DIR.glob("megalith-*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in {DATASET_DIR}")
    if not CAPTION_CSV.exists():
        raise RuntimeError(f"Missing caption CSV: {CAPTION_CSV}")

    con = connect()
    try:
        load_captions(con)
        for parquet_path in parquet_files:
            export_metadata(con, parquet_path)
    finally:
        con.close()


if __name__ == "__main__":
    main()
