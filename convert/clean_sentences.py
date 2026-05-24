#!/usr/bin/env python3
"""
Flatten and clean blip3-grounding-50m dataset.
- Flat destination directory: all tars renamed as  01___chunk_000___00000.tar
- Inside each tar, sentences with bad words are removed from .txt files
- All other files (parquet, json stats, _SUCCESS) are ignored
"""

import io
import multiprocessing as mp
import os
import re
import tarfile
import time
from pathlib import Path

SRC = Path("/path/to/data/vision-datasets/raw/stage2/hf___Salesforce___blip3-grounding-50m")
DST = Path("/path/to/data/vision-datasets/raw/stage2/hf___Salesforce___blip3-grounding-50m___cleaned")

BAD_WORDS = [
    "sentence",
    "grammar",
    "well-structured",
    "real-world information",
    "description",
    "structure",
]

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
BAD_RE = re.compile("|".join(re.escape(w) for w in BAD_WORDS), re.IGNORECASE)


def filter_txt(content: bytes) -> bytes:
    text = content.decode("utf-8", errors="replace").strip()
    sentences = SENT_SPLIT.split(text)
    kept = [s for s in sentences if not BAD_RE.search(s)]
    return " ".join(kept).encode("utf-8")


def process_tar(args):
    """Repack a single tar file with filtered .txt content."""
    src_tar, dst_tar = args
    if os.path.exists(dst_tar):
        return ("skip", src_tar)
    try:
        buf = io.BytesIO()
        with (
            tarfile.open(src_tar, "r") as src_tf,
            tarfile.open(fileobj=buf, mode="w") as dst_tf,
        ):
            for member in src_tf.getmembers():
                if not member.isfile():
                    dst_tf.addfile(member)
                    continue
                f = src_tf.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                if member.name.endswith(".txt"):
                    data = filter_txt(data)
                info = tarfile.TarInfo(name=member.name)
                info.size = len(data)
                info.mtime = member.mtime
                info.mode = member.mode
                dst_tf.addfile(info, io.BytesIO(data))

        # Write atomically
        tmp = dst_tar + ".tmp"
        with open(tmp, "wb") as out:
            out.write(buf.getvalue())
        os.rename(tmp, dst_tar)
        return ("ok", src_tar)
    except Exception as e:
        return ("err", src_tar, str(e))


def collect_jobs():
    jobs = []
    for subdir in sorted(SRC.iterdir()):
        if not subdir.is_dir():
            continue
        top = subdir.name  # e.g. '01'
        for chunk in sorted(subdir.iterdir()):
            if not chunk.is_dir() or not chunk.name.startswith("chunk_"):
                continue
            for f in sorted(chunk.iterdir()):
                if f.suffix == ".tar":
                    flat_name = f"{top}___{chunk.name}___{f.name}"
                    dst = str(DST / flat_name)
                    jobs.append((str(f), dst))
    return jobs


def main():
    num_workers = max(1, mp.cpu_count() - 4)
    print(f"Workers: {num_workers}")
    print("Collecting jobs...")
    t0 = time.time()

    DST.mkdir(parents=True, exist_ok=True)
    jobs = collect_jobs()
    print(f"  {len(jobs)} tar files to process")
    print(f"  (collected in {time.time() - t0:.1f}s)")

    done = errors = skipped = 0
    report_every = max(1, len(jobs) // 200)
    t_start = time.time()

    with mp.Pool(num_workers) as pool:
        for result in pool.imap_unordered(process_tar, jobs, chunksize=1):
            if result[0] == "ok":
                done += 1
            elif result[0] == "skip":
                skipped += 1
                done += 1
            else:
                errors += 1
                done += 1
                print(f"  ERROR: {result[1]}: {result[2]}", flush=True)

            if done % report_every == 0 or done == len(jobs):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(jobs) - done) / rate if rate > 0 else 0
                pct = 100 * done / len(jobs)
                print(
                    f"  [{pct:5.1f}%] {done}/{len(jobs)}  "
                    f"{rate:.1f} tars/s  ETA {eta / 60:.1f}m  "
                    f"errors={errors} skipped={skipped}",
                    flush=True,
                )

    print(f"\nDone in {(time.time() - t0) / 60:.1f}m — {errors} errors")


if __name__ == "__main__":
    main()
