#!/usr/bin/env python3
"""
Download Cooper Hewitt (chndm) images from the Smithsonian IDS service.

chndm has the richest design-object notes (⭐⭐⭐⭐) but no local images.
Images are fetched from:
  https://ids.si.edu/ids/deliveryService?id={idsId}&max=4096
Saved to:
  media/chndm/{idsId}.jpg

Rate-limited to ~4 requests/sec. Skips already-downloaded files.
Run this in parallel with 01_build_index.py — it is independent.
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Semaphore

import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW = Path("/path/to/data/vision-datasets/raw/cooldown/s3___smithsonian___OpenAccess")
META_DIR = RAW / "metadata/edan/chndm"
OUT_DIR = RAW / "media/chndm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = Path("/tmp/toolbox/smithsonian/data/download_chndm.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH),
    ],
)
log = logging.getLogger("download_chndm")

# ── Rate limiter ───────────────────────────────────────────────────────────────
RATE_PER_SEC = 4  # target requests/second
WORKERS = 4  # concurrent threads (each does ~1 req/sec → 4 req/sec total)
TIMEOUT = 30  # seconds per request
MAX_RETRIES = 3
IDS_BASE = "https://ids.si.edu/ids/deliveryService"
IDS_MAX_DIM = 4096  # request highest available resolution

# Simple inter-thread rate limiter using a time-bucket semaphore
_rate_sem = Semaphore(RATE_PER_SEC)


def _refill_rate_sem():
    """Refill the semaphore every second in a background thread."""
    import threading

    def _loop():
        while True:
            time.sleep(1.0)
            for _ in range(RATE_PER_SEC):
                try:
                    _rate_sem.release()
                except ValueError:
                    pass  # already at max

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


# ── Metadata scanning ──────────────────────────────────────────────────────────


def collect_idsIds() -> list[str]:
    """Scan all chndm metadata shards and collect unique idsIds."""
    seen = set()
    idsIds = []

    if not META_DIR.exists():
        log.error(f"chndm metadata directory not found: {META_DIR}")
        return []

    shards = [META_DIR / fn for fn in os.listdir(META_DIR) if fn != "index.txt" and os.path.getsize(META_DIR / fn) > 0]
    log.info(f"Scanning {len(shards)} metadata shards…")

    for shard_path in shards:
        try:
            with open(shard_path) as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        d = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    content = d.get("content", {})
                    dnr = content.get("descriptiveNonRepeating", {})
                    media_obj = dnr.get("online_media", {})
                    media_items = media_obj.get("media", []) if isinstance(media_obj, dict) else []
                    for m in media_items:
                        ids_id = m.get("idsId", "")
                        if ids_id and ids_id not in seen:
                            seen.add(ids_id)
                            idsIds.append(ids_id)
        except Exception as e:
            log.warning(f"Error reading {shard_path}: {e}")

    log.info(f"Found {len(idsIds):,} unique idsIds in chndm metadata")
    return idsIds


# ── Download worker ────────────────────────────────────────────────────────────


def download_one(session: requests.Session, ids_id: str) -> tuple[str, str]:
    """
    Download one image. Returns (ids_id, status) where status is one of:
    'ok', 'skip' (already exists), 'error:<msg>', '404'.
    """
    out_path = OUT_DIR / f"{ids_id}.jpg"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return ids_id, "skip"

    url = f"{IDS_BASE}?id={ids_id}&max={IDS_MAX_DIM}"

    for attempt in range(MAX_RETRIES):
        _rate_sem.acquire()  # block until rate budget available
        try:
            resp = session.get(url, timeout=TIMEOUT, allow_redirects=True)
            if resp.status_code == 200:
                content_type = resp.headers.get("Content-Type", "")
                if "image" not in content_type and len(resp.content) < 1000:
                    return ids_id, f"error:unexpected_content_type:{content_type}"
                out_path.write_bytes(resp.content)
                return ids_id, "ok"
            elif resp.status_code == 404:
                return ids_id, "404"
            elif resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            else:
                time.sleep(2)
                continue
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
            else:
                return ids_id, f"error:{e}"

    return ids_id, "error:max_retries"


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    idsIds = collect_idsIds()
    if not idsIds:
        log.error("No idsIds found. Exiting.")
        sys.exit(1)

    # Skip already downloaded
    to_download = [
        i for i in idsIds if not (OUT_DIR / f"{i}.jpg").exists() or (OUT_DIR / f"{i}.jpg").stat().st_size < 1000
    ]
    log.info(f"To download: {len(to_download):,}  (already present: {len(idsIds) - len(to_download):,})")

    if not to_download:
        log.info("All images already downloaded.")
        return

    eta_hours = len(to_download) / RATE_PER_SEC / 3600
    log.info(f"Estimated time at {RATE_PER_SEC}/sec: {eta_hours:.1f} hours")

    _refill_rate_sem()  # start background rate-limiter refill thread

    stats = {"ok": 0, "skip": 0, "404": 0, "error": 0}
    last_log = time.time()
    processed = 0

    # One requests.Session per worker thread
    def make_session():
        s = requests.Session()
        s.headers.update({"User-Agent": "SmithsonianDataset/1.0 (research)"})
        return s

    sessions = [make_session() for _ in range(WORKERS)]

    def worker(args):
        idx, ids_id = args
        session = sessions[idx % WORKERS]
        return download_one(session, ids_id)

    with ThreadPoolExecutor(max_workers=WORKERS) as exe:
        futures = {exe.submit(worker, (i, ids_id)): ids_id for i, ids_id in enumerate(to_download)}

        for future in as_completed(futures):
            ids_id = futures[future]
            try:
                _, status = future.result()
            except Exception as e:
                status = f"error:{e}"

            if status == "ok":
                stats["ok"] += 1
            elif status == "skip":
                stats["skip"] += 1
            elif status == "404":
                stats["404"] += 1
            else:
                stats["error"] += 1
                log.debug(f"Failed {ids_id}: {status}")

            processed += 1
            now = time.time()
            if now - last_log >= 60:
                pct = processed / len(to_download) * 100
                rate = stats["ok"] / max(1, now - _start)
                log.info(
                    f"Progress: {processed:,}/{len(to_download):,} ({pct:.1f}%) | "
                    f"ok={stats['ok']} 404={stats['404']} err={stats['error']} | "
                    f"{rate:.1f} img/s"
                )
                last_log = now

    log.info(f"\nDownload complete: {stats}")
    log.info(f"Images saved to: {OUT_DIR}")
    total_files = len(list(OUT_DIR.glob("*.jpg")))
    log.info(f"Total files in {OUT_DIR}: {total_files:,}")


_start = time.time()

if __name__ == "__main__":
    main()
