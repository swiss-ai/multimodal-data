#!/usr/bin/env python3
"""
Download arXiv PDFs from export.arxiv.org with rate limiting.

Usage:
    download_pdfs.py                    # download all from urls.txt
    download_pdfs.py --test N           # download first N URLs only
    download_pdfs.py --rate R           # requests/sec (default 4)
    download_pdfs.py --concurrency C    # parallel connections (default 32)
"""

import argparse
import asyncio
import time
from pathlib import Path

import aiofiles
import aiohttp
from tqdm import tqdm

URLS_FILE = Path(__file__).parent / "urls.txt"
OUT_DIR = Path(__file__).parent / "data" / "pdfs"

HEADERS = {"User-Agent": "Apertus/1.0 (research data preparation)"}
CHUNK = 256 * 1024  # 256 KB streaming chunk


class TokenBucket:
    """Async token bucket for rate limiting (serialised via lock)."""

    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = 0.0  # start empty — no initial burst
        self._lock = asyncio.Lock()
        self._last = time.monotonic()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            self.tokens = min(self.rate, self.tokens + (now - self._last) * self.rate)
            self._last = now
            if self.tokens < 1.0:
                await asyncio.sleep((1.0 - self.tokens) / self.rate)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0


async def download_one(
    session: aiohttp.ClientSession,
    url: str,
    filename: str,
    out_dir: Path,
    bucket: TokenBucket,
    retries: int = 5,
) -> str:
    """Stream one PDF to disk. Returns 'ok', 'skip', or 'error'."""
    dest = out_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"

    tmp = dest.with_suffix(".tmp")
    for attempt in range(retries):
        await bucket.acquire()
        try:
            async with session.get(url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status == 200:
                    async with aiofiles.open(tmp, "wb") as f:
                        async for chunk in resp.content.iter_chunked(CHUNK):
                            await f.write(chunk)
                    # Verify it's an actual PDF before committing
                    if tmp.stat().st_size == 0:
                        tmp.unlink(missing_ok=True)
                        return "error"
                    tmp.rename(dest)
                    return "ok"
                elif resp.status in (429, 503):
                    await asyncio.sleep(10 * (2**attempt))
                elif resp.status == 404:
                    tmp.unlink(missing_ok=True)
                    return "error"
                else:
                    await asyncio.sleep(2**attempt)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            tmp.unlink(missing_ok=True)
            if attempt < retries - 1:
                await asyncio.sleep(3 * (2**attempt))

    tmp.unlink(missing_ok=True)
    return "error"


async def worker(
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    out_dir: Path,
    bucket: TokenBucket,
    counters: dict,
    bar: tqdm,
):
    """Worker that pulls (url, filename) from queue until empty."""
    while True:
        try:
            url, fname = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        result = await download_one(session, url, fname, out_dir, bucket)
        counters[result] = counters.get(result, 0) + 1
        bar.update(1)
        queue.task_done()


async def run(urls: list[tuple[str, str]], rate: float, concurrency: int):
    queue: asyncio.Queue = asyncio.Queue()
    for item in urls:
        queue.put_nowait(item)

    bucket = TokenBucket(rate)
    counters: dict = {}

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        with tqdm(total=len(urls), desc="PDFs", unit="pdf") as bar:
            workers = [
                asyncio.create_task(worker(queue, session, OUT_DIR, bucket, counters, bar)) for _ in range(concurrency)
            ]
            await asyncio.gather(*workers)

    ok = counters.get("ok", 0)
    skip = counters.get("skip", 0)
    error = counters.get("error", 0)
    print(f"\nDone: ok={ok}  skipped={skip}  errors={error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0, help="Download only first N URLs")
    parser.add_argument("--rate", type=float, default=4.0, help="Requests/sec (default 4)")
    parser.add_argument("--concurrency", type=int, default=32, help="Parallel connections (default 32)")
    args = parser.parse_args()

    lines = URLS_FILE.read_text().splitlines()
    urls = [(p[0], p[1]) for line in lines if (p := line.strip().split("\t")) and len(p) == 2]

    if args.test:
        urls = urls[: args.test]
        print(f"TEST MODE: {len(urls)} PDFs")
    else:
        already = sum(1 for _, fname in urls if (OUT_DIR / fname).exists())
        print(
            f"Downloading {len(urls)} PDFs ({already} already done, {len(urls) - already} remaining) at {args.rate} req/s"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(urls, args.rate, args.concurrency))


if __name__ == "__main__":
    main()
