#!/usr/bin/env python3
"""Download a CommonVoice 25.0 locale and emit per-split parquet without
ever extracting the ``clips/`` directory to disk.

Pipeline:
  1. Mozilla Data Collective API → temporary download URL
  2. ``aria2c`` multi-connection download → single ``.tar.gz`` on scratch
  3. Single stream pass through the tar (pigz-decompressed when available;
     producer/consumer threads run gzip-decompress and parquet-write
     concurrently):
       - First ~11 members are TSVs (verified empirically): parse all 6
         split TSVs into memory and build a ``clip_filename → primary_split``
         map plus a per-clip metadata dict (column-merged across splits).
       - Remaining members are ``clips/*.mp3``: route each to the right
         per-split parquet writer with the stored metadata + raw mp3 bytes.
  4. Close all writers, delete tar (or ``--keep-tar`` for debug).

Per-clip primary-split priority (each clip lands in exactly one parquet):

    train  >  dev  >  test  >  validated_extra  >  other  >  invalidated

``validated_extra`` = clips in ``validated.tsv`` but not in train/dev/test
(spillover from the official train cap + per-speaker stratification cuts;
Mozilla doesn't materialize it as a named split, so we compute it).

Usage::

    # Single locale
    export MOZILLA_DC_TOKEN_FILE=$HOME/.mozilla_dc_token  # default
    python download_cv_to_parquet.py --lang zh-CN \\
        --output-dir /capstor/store/.../PARQUET/commonvoice25/zh-CN

    # Multiple locales sequentially
    for lang in zh-CN ka ru th; do
        python download_cv_to_parquet.py --lang "$lang" \\
            --output-dir "/capstor/store/.../PARQUET/commonvoice25/$lang"
    done

Already-downloaded tarballs can be reused with ``--skip-download``::

    python download_cv_to_parquet.py --lang zh-TW --skip-download \\
        --tar-path /iopsstor/scratch/.../cv25_tmp/cv25_zh-TW.tar.gz \\
        --output-dir /capstor/.../PARQUET/commonvoice25/zh-TW
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mozilla Data Collective dataset IDs — Common Voice 25.0 (2026-03-09).
#
# Fetched from https://mozilladatacollective.com/organization/cmfh0j9o10006ns07jq45h7xk
# v25 covers 290 locales; the subset below mirrors the v24 lang list used
# by download_commonvoice24.sh so existing dataset cards stay aligned.
# Add new locales here on demand (e.g. ``gsw`` Swiss German is now
# available: ``cmn29hk3a0184mm072ledt03r``).
# ---------------------------------------------------------------------------

DATASETS: dict[str, str] = {
    "ar": "cmn2g7uu701fqo1072r5na25l",
    "sq": "cmn29zkso01aimm07wb1ar40j",
    "hy-AM": "cmn2e8k9z01kymm07yqqy4bk1",
    "eu": "cmn2hwe0d01n8mm07wug9r5he",
    "gsw": "cmn29hk3a0184mm072ledt03r",  # Swiss German / Alsatian — new in v25
    "kk": "cmn29sufc018ymm071hvsk595",   # Kazakh — verified from Mozilla DC v25 catalog
    # --- Indian subcontinent (top-population) ---
    "hi": "cmn2cxzy701iumm077t5ayw0e",    # Hindi
    "bn": "cmn3ipo8b00ejmi079e8upl2k",    # Bengali
    "ur": "cmn2h58bw01mwmm07t3ypteqz",    # Urdu
    "ta": "cmn2gfvyp01geo107izoftfki",    # Tamil
    "te": "cmn29lt270168o107nhemmxkh",    # Telugu
    "mr": "cmn2cxubn01ebo107piypnoix",    # Marathi
    "be": "cmn4xg3a900d3nu075gnh4jpt",
    "bg": "cmn2coplj01gqmm07dbibr68y",
    "ca": "cmnd4la5a02fwmh074t1fx5y9",
    "cs": "cmn2h5zd801h3o1075tita1ap",
    "da": "cmn2cptsh01hymm07mulngxv0",
    "nl": "cmn2g7nu901fmo107a1ydn0n5",
    "en": "cmndapwry02jnmh07dyo46mot",
    "eo": "cmn4o8691005pnu07fxmq06px",
    "et": "cmn2e880l01kumm07i9upoz99",
    "fi": "cmn2cyal501jamm07q2dnsy5x",
    "fr": "cmn5zugst00w3nv07upovf2bg",
    "gl": "cmn2h0nw001momm07xxarkyd4",
    "ka": "cmn2h4m7901gzo1072qn7zoes",
    "de": "cmn4rsdh6009unz07jdn2ol9p",
    "el": "cmn2cx91x01dno10754vxfu3b",
    "hu": "cmn2g9aoi01fyo107xhdrwb5d",
    "is": "cmn1q47o300vjo1078y9lpkff",
    "ga-IE": "cmn2cp9uy01hemm07jfogi1zf",
    "it": "cmn2h0yei01msmm07u8z5vu87",
    "ja": "cmn2hm68r01n4mm071qux43yu",
    "ko": "cmn2ale8p018bo107nbvos0f7",
    "lv": "cmn2gmdle01gmo1071lmpg5sj",
    "lt": "cmn2cxca301dro107ucz5j8ey",
    "mk": "cmn2e8yb101lemm07j3flmvjs",
    "mt": "cmn2cyjd001jmmm07meob0o2j",
    "nb-NO": "cmn29lkh2018kmm07ywneb3o0",
    "nn-NO": "cmn29l17i015oo107fb396rv4",
    "pl": "cmn27nz69015hmm0720txf781",
    "pt": "cmn29f4cb017bmm07pd9yd8mw",
    "ro": "cmn2e8rmi01l6mm07vxurptse",
    "ru": "cmn2h1dg201gro107lpynbbd6",
    "sr": "cmn2bjv8101aho107xj4uoklm",
    "sk": "cmn2e8ojy01l2mm07giwrvaqf",
    "sl": "cmn2cy7z701j6mm07axskhd0a",
    "es": "cmn4z1n52000knv07h01532dd",
    "sv-SE": "cmn2e8gr301evo1079gujuzqr",
    "tr": "cmn2e7kbl01k2mm07gm5n1bc9",
    "uk": "cmn2e7qgt01kamm07oftersjt",
    "cy": "cmn2g9w1h01m6mm07lq2w14dd",
    "zh-CN": "cmn3iaztg00e4mb070uvufz7q",
    "zh-HK": "cmn2g8zqd01m2mm07prcmehku",
    "zh-TW": "cmn2g7eaj01fio10769r1m96n",
    "rm-sursilv": "cmn2cq76201iimm07avtwokjf",
    "th": "cmn2h1svx01gvo1074l8g2a27",
    "vi": "cmn2cojzk01gimm07zqlmypmn",
    "yue": "cmn29rqn9016to107eniyak65",
    "id": "cmn2e8ats01eno107glwgoasv",
    "fa": "cmn2gho8i01gio107ckfuqzxo",
}

# Mozilla migrated the API host between v24 and v25 releases. The old
# datacollective.mozillafoundation.org/api now 301-redirects, but the
# Authorization header is stripped on cross-host redirects, so callers
# must hit the new host directly.
API_BASE = "https://mozilladatacollective.com/api"
DEFAULT_CV_RELEASE = "cv-corpus-25.0-2026-03-09"

ALL_TSV_SPLITS = ("train", "dev", "test", "other", "validated", "invalidated")
PARQUET_SPLITS = ("train", "dev", "test", "validated_extra", "other", "invalidated")
PRIMARY_PRIORITY = ("train", "dev", "test", "validated", "other", "invalidated")
PARQUET_BATCH_ROWS = 256  # rows per RecordBatch flush
QUEUE_MAX_SIZE = 1024     # bounded queue between reader and writer threads


# ---------------------------------------------------------------------------
# Mozilla DC API + aria2c download
# ---------------------------------------------------------------------------


def get_download_url(token: str, dataset_id: str) -> str:
    response = requests.post(
        f"{API_BASE}/datasets/{dataset_id}/download",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    url = payload.get("downloadUrl")
    if not url:
        raise RuntimeError(f"Mozilla DC returned no downloadUrl: {payload!r}")
    return url


def download_with_aria2c(url: str, dst: Path, *, connections: int = 16) -> None:
    """Download *url* to *dst*. Uses aria2c (multi-connection) when available,
    else falls back to curl (single-connection but always available in
    container envs that lack aria2c)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("aria2c") is not None:
        cmd = [
            "aria2c",
            "-x", str(connections),
            "-s", str(connections),
            "--continue=true",
            "--max-tries=5",
            "--retry-wait=10",
            "--summary-interval=30",
            "-d", str(dst.parent),
            "-o", dst.name,
            url,
        ]
        logger.info("aria2c download: %s", dst)
        subprocess.run(cmd, check=True)
        return
    if shutil.which("curl") is not None:
        # Single-connection fallback. Cloudflare R2 sustains ~200-500 MB/s
        # on a single TCP stream, so this is fine for ~tens-of-GB tarballs.
        # `--retry` handles transient drops; `-C -` resumes from partial.
        logger.info("aria2c not available, using curl single-connection: %s", dst)
        cmd = [
            "curl", "-sS", "-L",
            "--retry", "5",
            "--retry-delay", "10",
            "-C", "-",
            "--output", str(dst),
            url,
        ]
        subprocess.run(cmd, check=True)
        return
    raise RuntimeError(
        "Neither aria2c nor curl is available. Install one of them or use a "
        "container that includes them (e.g. nemo_25_11 has curl)."
    )


# ---------------------------------------------------------------------------
# Single-pass tar streaming
# ---------------------------------------------------------------------------


def _resolve_tsv_split(member_name: str) -> str | None:
    """Return one of ALL_TSV_SPLITS if *member_name* is a recognized TSV."""
    base = member_name.rsplit("/", 1)[-1]
    if not base.endswith(".tsv"):
        return None
    stem = base[:-4]
    return stem if stem in ALL_TSV_SPLITS else None


def _is_clip_member(member: tarfile.TarInfo) -> bool:
    return member.isfile() and member.name.endswith(".mp3") and "/clips/" in member.name


def _read_tsv_to_rows(content: bytes) -> tuple[list[dict], list[str]]:
    text = content.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE)
    fields = reader.fieldnames or []
    return list(reader), fields


def _assign_primary(clip: str, split_sets: dict[str, set[str]]) -> str | None:
    for split in PRIMARY_PRIORITY:
        if clip in split_sets[split]:
            return "validated_extra" if split == "validated" else split
    return None


@contextmanager
def _open_tar_stream(tar_path: Path, *, use_pigz: bool) -> Iterator[tarfile.TarFile]:
    """Open *tar_path* as a streaming TarFile.

    When *use_pigz* is True and ``pigz`` is on PATH, decompression runs in a
    subprocess (pigz uses internal threads for inflate + CRC, which is faster
    than Python's single-threaded zlib). Otherwise falls back to tarfile's
    built-in gzip mode.
    """
    if use_pigz and shutil.which("pigz") is not None:
        proc = subprocess.Popen(
            ["pigz", "-dc", str(tar_path)],
            stdout=subprocess.PIPE,
            bufsize=8 * 1024 * 1024,
        )
        assert proc.stdout is not None
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                yield tar
            proc.stdout.close()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"pigz exited with status {rc} for {tar_path}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
        return

    with tarfile.open(tar_path, mode="r|gz") as tar:
        yield tar


class _ShardingParquetWriter:
    """Per-split parquet writer with optional row-count-based sharding.

    When *shard_rows* is None or 0, writes a single ``<split>.parquet`` file.
    Otherwise rotates to ``<split>/part-NNNNN.parquet`` every *shard_rows* rows.
    """

    def __init__(
        self,
        output_dir: Path,
        split: str,
        schema: pa.Schema,
        *,
        compression: str | None,
        shard_rows: int | None,
    ) -> None:
        self.output_dir = output_dir
        self.split = split
        self.schema = schema
        self.compression = compression
        self.shard_rows = shard_rows or 0
        self._writer: pq.ParquetWriter | None = None
        self._part_index = 0
        self._rows_in_part = 0
        if self.shard_rows > 0:
            (output_dir / split).mkdir(parents=True, exist_ok=True)

    def _open_part(self) -> pq.ParquetWriter:
        if self.shard_rows > 0:
            path = self.output_dir / self.split / f"part-{self._part_index:05d}.parquet"
        else:
            path = self.output_dir / f"{self.split}.parquet"
        return pq.ParquetWriter(str(path), self.schema, compression=self.compression)

    def write_batch(self, batch: pa.RecordBatch) -> None:
        if self._writer is None:
            self._writer = self._open_part()
            self._rows_in_part = 0
        self._writer.write_batch(batch)
        self._rows_in_part += batch.num_rows
        if self.shard_rows > 0 and self._rows_in_part >= self.shard_rows:
            self._writer.close()
            self._writer = None
            self._part_index += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _build_schema(seen_columns: set[str]) -> tuple[pa.Schema, list[str]]:
    """Finalize the parquet schema after all TSVs are parsed.

    Drops ``path`` (we expose it as ``clip_id``); appends ``clip_id``, ``split``,
    ``audio_bytes``. Returns (schema, ordered list of TSV column names).
    """
    added_names = {"clip_id", "split", "audio_bytes"}
    tsv_cols = sorted(c for c in seen_columns if c != "path" and c not in added_names)
    fields = [pa.field(c, pa.string()) for c in tsv_cols]
    fields += [
        pa.field("clip_id", pa.string()),
        pa.field("split", pa.string()),
        pa.field("audio_bytes", pa.binary()),
    ]
    return pa.schema(fields), tsv_cols


# Sentinel objects for the producer/consumer queue protocol.
_SCHEMA_READY = object()  # signals schema is set in shared state
_END_OF_STREAM = object()


def stream_tar_to_parquet(
    tar_path: Path,
    output_dir: Path,
    *,
    locale: str,
    parquet_compression: str | None = "zstd",
    use_pigz: bool = True,
    use_concurrent: bool = True,
    shard_rows: int | None = None,
) -> dict[str, dict]:
    """Single-pass tar stream → per-split parquet.

    Optimization knobs (defaults are all-on):
      - ``use_pigz``: route gzip decompression through a pigz subprocess
        (parallel inflate + CRC). Falls back to Python's zlib if pigz is
        not available.
      - ``use_concurrent``: run tar reading + parquet writing on separate
        threads. zlib (or pigz I/O) and zstd both release the GIL during
        their C-level work, so the two threads make real progress concurrently.
      - ``shard_rows``: when set, rotate per-split output to
        ``<split>/part-NNNNN.parquet`` every N rows. Useful for big locales
        where a single parquet would exceed ~2 GB.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not use_concurrent:
        return _stream_serial(
            tar_path, output_dir,
            locale=locale,
            parquet_compression=parquet_compression,
            use_pigz=use_pigz,
            shard_rows=shard_rows,
        )

    # Producer/consumer:
    #   - Producer (worker thread) reads the tar, parses TSVs synchronously,
    #     then queues (clip, split, audio_bytes) tuples for each mp3.
    #   - Consumer (this thread) drains the queue, builds rows, batches them
    #     per split, and writes to parquet (with optional sharding).
    work_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
    shared: dict = {
        "schema": None,
        "tsv_columns": None,
        "metadata": None,
        "split_counts": None,
        "error": None,
    }
    schema_ready = threading.Event()

    def producer() -> None:
        try:
            split_sets: dict[str, set[str]] = {s: set() for s in ALL_TSV_SPLITS}
            metadata: dict[str, dict] = {}
            seen_columns: set[str] = set()
            tsv_done = False

            with _open_tar_stream(tar_path, use_pigz=use_pigz) as tar:
                for member in tar:
                    if member.isdir():
                        continue

                    tsv_split = _resolve_tsv_split(member.name)
                    if tsv_split is not None:
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        rows, fields = _read_tsv_to_rows(f.read())
                        seen_columns.update(fields)
                        for row in rows:
                            clip = row.get("path") or ""
                            if not clip:
                                continue
                            split_sets[tsv_split].add(clip)
                            metadata.setdefault(clip, {}).update(
                                {k: ("" if v is None else v) for k, v in row.items() if k != "path"}
                            )
                        continue

                    # Non-split-TSV members carry no routing info; skip.
                    if not _is_clip_member(member):
                        continue

                    if not tsv_done:
                        tsv_done = True
                        schema, tsv_cols = _build_schema(seen_columns)
                        shared["schema"] = schema
                        shared["tsv_columns"] = tsv_cols
                        shared["metadata"] = metadata
                        shared["split_counts"] = {s: len(v) for s, v in split_sets.items()}
                        schema_ready.set()
                        logger.info(
                            "TSV pre-load complete: %d unique clips, columns=%s",
                            len(metadata), tsv_cols,
                        )
                        logger.info("Per-split TSV row counts: %s", shared["split_counts"])

                    clip = member.name.rsplit("/", 1)[-1]
                    split = _assign_primary(clip, split_sets)
                    if split is None:
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    work_queue.put((clip, split, f.read()))
        except BaseException as exc:  # noqa: BLE001
            shared["error"] = exc
        finally:
            schema_ready.set()  # release consumer if producer aborted before mp3s
            work_queue.put(_END_OF_STREAM)

    started = time.monotonic()
    reader = threading.Thread(target=producer, name="cv24-tar-reader", daemon=True)
    reader.start()
    schema_ready.wait()

    if shared["error"] is not None:
        reader.join()
        raise shared["error"]

    schema = shared["schema"]
    tsv_columns = shared["tsv_columns"]
    metadata = shared["metadata"]

    writers = {
        s: _ShardingParquetWriter(
            output_dir, s, schema,
            compression=parquet_compression, shard_rows=shard_rows,
        )
        for s in PARQUET_SPLITS
    }
    buffers: dict[str, list[dict]] = {s: [] for s in PARQUET_SPLITS}
    counts: dict[str, dict] = {s: {"rows": 0, "bytes": 0} for s in PARQUET_SPLITS}

    def _flush(split: str) -> None:
        rows = buffers[split]
        if not rows:
            return
        batch = pa.RecordBatch.from_pylist(rows, schema=schema)
        writers[split].write_batch(batch)
        buffers[split] = []

    while True:
        item = work_queue.get()
        if item is _END_OF_STREAM:
            break
        clip, split, audio_bytes = item
        base_meta = metadata.get(clip, {})
        row = {col: base_meta.get(col, "") for col in tsv_columns}
        row["locale"] = locale
        row["clip_id"] = clip
        row["split"] = split
        row["audio_bytes"] = audio_bytes
        buffers[split].append(row)
        counts[split]["rows"] += 1
        counts[split]["bytes"] += len(audio_bytes)
        if len(buffers[split]) >= PARQUET_BATCH_ROWS:
            _flush(split)

    for s in PARQUET_SPLITS:
        _flush(s)
    for w in writers.values():
        w.close()

    reader.join()
    if shared["error"] is not None:
        raise shared["error"]

    logger.info("Stream-pass finished in %.1fs", time.monotonic() - started)
    return counts


def _stream_serial(
    tar_path: Path,
    output_dir: Path,
    *,
    locale: str,
    parquet_compression: str | None,
    use_pigz: bool,
    shard_rows: int | None,
) -> dict[str, dict]:
    """Single-threaded fallback: keeps the tar reader and parquet writer
    on the same thread. Used when ``use_concurrent=False`` for debugging /
    determinism."""
    split_sets: dict[str, set[str]] = {s: set() for s in ALL_TSV_SPLITS}
    metadata: dict[str, dict] = {}
    seen_columns: set[str] = set()
    tsv_done = False
    schema: pa.Schema | None = None
    tsv_columns: list[str] = []
    writers: dict[str, _ShardingParquetWriter] = {}
    buffers: dict[str, list[dict]] = {s: [] for s in PARQUET_SPLITS}
    counts: dict[str, dict] = {s: {"rows": 0, "bytes": 0} for s in PARQUET_SPLITS}

    def _flush(split: str) -> None:
        rows = buffers[split]
        if not rows or schema is None:
            return
        if split not in writers:
            writers[split] = _ShardingParquetWriter(
                output_dir, split, schema,
                compression=parquet_compression, shard_rows=shard_rows,
            )
        batch = pa.RecordBatch.from_pylist(rows, schema=schema)
        writers[split].write_batch(batch)
        buffers[split] = []

    started = time.monotonic()
    with _open_tar_stream(tar_path, use_pigz=use_pigz) as tar:
        for member in tar:
            if member.isdir():
                continue

            tsv_split = _resolve_tsv_split(member.name)
            if tsv_split is not None:
                f = tar.extractfile(member)
                if f is None:
                    continue
                rows, fields = _read_tsv_to_rows(f.read())
                seen_columns.update(fields)
                for row in rows:
                    clip = row.get("path") or ""
                    if not clip:
                        continue
                    split_sets[tsv_split].add(clip)
                    metadata.setdefault(clip, {}).update(
                        {k: ("" if v is None else v) for k, v in row.items() if k != "path"}
                    )
                continue

            if not _is_clip_member(member):
                continue

            if not tsv_done:
                tsv_done = True
                schema, tsv_columns = _build_schema(seen_columns)
                logger.info(
                    "TSV pre-load complete: %d unique clips, columns=%s",
                    len(metadata), tsv_columns,
                )
                logger.info(
                    "Per-split TSV row counts: %s",
                    {s: len(v) for s, v in split_sets.items()},
                )

            clip = member.name.rsplit("/", 1)[-1]
            split = _assign_primary(clip, split_sets)
            if split is None:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            audio_bytes = f.read()

            base_meta = metadata.get(clip, {})
            row = {col: base_meta.get(col, "") for col in tsv_columns}
            row["locale"] = locale
            row["clip_id"] = clip
            row["split"] = split
            row["audio_bytes"] = audio_bytes
            buffers[split].append(row)
            counts[split]["rows"] += 1
            counts[split]["bytes"] += len(audio_bytes)
            if len(buffers[split]) >= PARQUET_BATCH_ROWS:
                _flush(split)

    for s in PARQUET_SPLITS:
        _flush(s)
    for w in writers.values():
        w.close()

    logger.info("Stream-pass finished in %.1fs (serial)", time.monotonic() - started)
    return counts


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------


def _split_output_paths(output_dir: Path, split: str, sharded: bool) -> list[str]:
    """Return the relative parquet path(s) for a split, post-write."""
    if sharded:
        split_dir = output_dir / split
        if not split_dir.is_dir():
            return []
        return sorted(f"{split}/{p.name}" for p in split_dir.glob("part-*.parquet"))
    single = output_dir / f"{split}.parquet"
    return [f"{split}.parquet"] if single.is_file() else []


def write_manifest(
    output_dir: Path,
    *,
    locale: str,
    cv_release: str,
    counts: dict[str, dict],
    tar_path: Path | None,
    sharded: bool,
) -> Path:
    manifest = {
        "schema_version": 1,
        "cv_release": cv_release,
        "locale": locale,
        "produced_by": "download_cv24_to_parquet.py",
        "splits": {
            split: {
                "rows": counts[split]["rows"],
                "audio_bytes_total": counts[split]["bytes"],
                "parquet_paths": _split_output_paths(output_dir, split, sharded),
            }
            for split in PARQUET_SPLITS
        },
        "totals": {
            "rows": sum(c["rows"] for c in counts.values()),
            "audio_bytes_total": sum(c["bytes"] for c in counts.values()),
        },
        "source_tar": str(tar_path) if tar_path else None,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", required=True, help=f"CV24 locale code (one of: {sorted(DATASETS)})")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where the per-split parquet + manifest.json land.")
    parser.add_argument("--tar-path", type=Path, default=None,
                        help="Where to download the tarball. Default: "
                             "/iopsstor/scratch/cscs/$USER/cv24_tmp/<lang>.tar.gz")
    parser.add_argument("--token-file", type=Path,
                        default=Path(
                            os.environ.get("MOZILLA_DC_TOKEN_FILE")
                            or f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}/.mozilla_dc_token"
                        ))
    parser.add_argument("--connections", type=int, default=16,
                        help="aria2c parallel connections per file (default: 16)")
    parser.add_argument("--cv-release", default=DEFAULT_CV_RELEASE,
                        help=f"CV release tag for manifest provenance (default: {DEFAULT_CV_RELEASE})")
    parser.add_argument("--parquet-compression", default="zstd",
                        choices=("zstd", "snappy", "gzip", "none"))
    parser.add_argument("--skip-download", action="store_true",
                        help="Use --tar-path as-is; do not contact Mozilla DC.")
    parser.add_argument("--keep-tar", action="store_true",
                        help="Leave the tarball on disk after parquet emission.")
    parser.add_argument("--no-pigz", action="store_true",
                        help="Force Python's zlib for gzip decompression even when pigz is available.")
    parser.add_argument("--no-concurrent", action="store_true",
                        help="Run tar reader and parquet writer on the same thread (slower; for debugging).")
    parser.add_argument("--shard-rows", type=int, default=0,
                        help="When > 0, write each split as <split>/part-NNNNN.parquet rotating "
                             "every N rows. Default 0 emits a single <split>.parquet per split.")
    args = parser.parse_args(argv)

    if args.lang not in DATASETS:
        parser.error(f"Unknown locale {args.lang!r}; supported: {sorted(DATASETS)}")

    if args.parquet_compression == "none":
        args.parquet_compression = None  # type: ignore[assignment]

    tar_path = args.tar_path
    if tar_path is None:
        scratch = Path(f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}/cv24_tmp")
        scratch.mkdir(parents=True, exist_ok=True)
        tar_path = scratch / f"commonvoice24_{args.lang}.tar.gz"

    if not args.skip_download:
        if not args.token_file.is_file():
            parser.error(
                f"Token file not found: {args.token_file}. "
                "Create with: echo YOUR_TOKEN > <path> && chmod 600 <path>"
            )
        token = args.token_file.read_text().strip()
        url = get_download_url(token, DATASETS[args.lang])
        download_with_aria2c(url, tar_path, connections=args.connections)
    elif not tar_path.is_file():
        parser.error(f"--skip-download set but --tar-path {tar_path} not found")

    counts = stream_tar_to_parquet(
        tar_path,
        args.output_dir,
        locale=args.lang,
        parquet_compression=args.parquet_compression,
        use_pigz=not args.no_pigz,
        use_concurrent=not args.no_concurrent,
        shard_rows=args.shard_rows or None,
    )

    manifest_path = write_manifest(
        args.output_dir,
        locale=args.lang,
        cv_release=args.cv_release,
        counts=counts,
        tar_path=tar_path if args.keep_tar else None,
        sharded=bool(args.shard_rows),
    )

    if not args.keep_tar and tar_path.is_file() and not args.skip_download:
        tar_path.unlink()
        logger.info("Removed tar: %s", tar_path)

    logger.info("\n=== %s parquet summary ===", args.lang)
    total_rows = 0
    total_bytes = 0
    for split in PARQUET_SPLITS:
        c = counts[split]
        if c["rows"]:
            logger.info(
                "  %-16s %10d rows  %10.2f MB audio",
                split, c["rows"], c["bytes"] / 1e6,
            )
            total_rows += c["rows"]
            total_bytes += c["bytes"]
    logger.info("  %-16s %10d rows  %10.2f MB audio", "TOTAL", total_rows, total_bytes / 1e6)
    logger.info("Manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
