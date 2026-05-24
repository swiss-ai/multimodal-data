"""
Extract frames from ego4d full-scale videos using narrations.parquet.
Designed to run as a SLURM job array (see submit_extract.sh).

For each parquet row the script writes:
  {OUT_DIR}/{video_uid}/{key}.img1.jpg          (always)
  {OUT_DIR}/{video_uid}/{key}.img2.jpg          (summaries with 5 timestamps only)
  {OUT_DIR}/{video_uid}/{key}.img3.jpg          (summaries with 5 timestamps only)
  {OUT_DIR}/{video_uid}/{key}.img4.jpg          (summaries with 5 timestamps only)
  {OUT_DIR}/{video_uid}/{key}.img5.jpg          (summaries with 5 timestamps only)
  {OUT_DIR}/{video_uid}/{key}.txt               caption with <|imgN|> placeholders

Caption format:
  <|img1|>
 [<|img2|>
  ...
  <|img5|>]
  {caption text}

Parallelism: each SLURM job handles a strided slice of videos (~48 per job for
201 jobs). Within a job, up to WORKERS videos are processed in parallel via
threads (safe because we only call subprocesses — no GIL contention).
Idempotent: already-complete samples (all images + txt exist) are skipped.

Alignment: ego4d.json is loaded to:
  - Snap timestamps to exact frame boundaries (all videos are 30 fps VP9).
  - Skip samples whose timestamps fall inside redacted intervals.
  - Use a two-step ffmpeg seek (fast pre-seek + short accurate seek) to avoid
    the VP9 keyframe-interval inaccuracy of a single fast seek.
"""

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import orjson
import polars as pl

PARQUET = "/tmp/ego4d/narrations.parquet"
VIDEO_DIR = Path("/path/to/data/vision-datasets/ego4d/v2/full_scale")
OUT_DIR = Path("/tmp/ego4d/narrations")
EGO4D_JSON = "/path/to/data/vision-datasets/ego4d/ego4d.json"

WORKERS = 32
PRE_SEEK_SEC = 4.0  # fast pre-seek this many seconds before target


# ── ego4d metadata ────────────────────────────────────────────────────────────


def load_video_meta(path: str) -> dict:
    """
    Returns {video_uid: {"fps": float, "redacted": [(start, end), ...]}}
    from ego4d.json.
    """
    print(f"Loading {path} …", flush=True)
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    meta = {}
    for v in data.get("videos", []):
        uid = v["video_uid"]
        fps = float(v.get("video_metadata", {}).get("fps", 30.0))
        redacted = [(float(r["start_sec"]), float(r["end_sec"])) for r in v.get("redacted_intervals", [])]
        meta[uid] = {"fps": fps, "redacted": redacted}
    return meta


def snap_to_frame(ts: float, fps: float) -> float:
    """Round timestamp to the nearest frame boundary."""
    return round(ts * fps) / fps


def is_redacted(ts: float, redacted: list) -> bool:
    """Return True if ts falls inside any redacted interval."""
    return any(start <= ts <= end for start, end in redacted)


# ── Frame extraction ──────────────────────────────────────────────────────────


def extract_frame(video_path: str, ts: float, out_path: str, max_dim: int) -> bool:
    """
    Two-step seek: fast-seek to (ts - PRE_SEEK_SEC), then accurate seek
    PRE_SEEK_SEC forward. This avoids VP9 keyframe-interval misalignment
    while keeping decode time short.
    """
    pre = max(0.0, ts - PRE_SEEK_SEC)
    fine = ts - pre  # always == min(PRE_SEEK_SEC, ts)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{pre:.3f}",  # fast pre-seek (keyframe)
        "-i",
        video_path,
        "-ss",
        f"{fine:.3f}",  # accurate fine-seek (decode)
        "-frames:v",
        "1",
        "-vf",
        f"scale={max_dim}:{max_dim}:force_original_aspect_ratio=decrease",
        "-q:v",
        "2",  # JPEG quality (2 = high, 31 = low)
        out_path,
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


# ── Per-video worker ──────────────────────────────────────────────────────────


def process_video(video_uid: str, rows, out_base: Path, vmeta: dict):
    """Extract all frames for one video. Returns (done, skipped, errors)."""
    video_path = str(VIDEO_DIR / f"{video_uid}.mp4")
    if not os.path.exists(video_path):
        print(f"  [MISS] {video_uid}.mp4", flush=True)
        return 0, 0, len(rows)

    meta = vmeta.get(video_uid, {})
    fps = meta.get("fps", 30.0)
    redacted = meta.get("redacted", [])

    video_dir = out_base / video_uid
    video_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = errors = 0

    for row in rows:
        key = row["key"]
        timestamps = row["timestamps"]
        caption = row["caption"]
        n_imgs = len(timestamps)
        max_dim = 768

        # Snap to frame boundaries
        timestamps = [snap_to_frame(ts, fps) for ts in timestamps]

        # Skip samples with any timestamp in a redacted interval
        if redacted and any(is_redacted(ts, redacted) for ts in timestamps):
            skipped += 1
            continue

        img_paths = [str(video_dir / f"{key}.img{i + 1}.jpg") for i in range(n_imgs)]
        txt_path = video_dir / f"{key}.txt"

        # Resume: skip if fully written
        if txt_path.exists() and all(os.path.exists(p) for p in img_paths):
            skipped += 1
            continue

        # Extract each frame
        ok = True
        for i, (ts, img_path) in enumerate(zip(timestamps, img_paths)):
            if not extract_frame(video_path, ts, img_path, max_dim):
                print(f"  [ERR]  {key} img{i + 1} ts={ts:.2f}", flush=True)
                ok = False
                break

        if not ok:
            errors += 1
            continue

        # Write caption
        placeholders = "\n".join(f"<|img{i + 1}|>" for i in range(n_imgs))
        txt_path.write_text(f"{placeholders}\n{caption}")
        done += 1

    return done, skipped, errors


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--num-jobs", type=int, required=True)
    args = parser.parse_args()

    vmeta = load_video_meta(EGO4D_JSON)

    df = pl.read_parquet(PARQUET)

    all_videos = df["video_uid"].unique().sort().to_list()
    # all_videos = all_videos[:5]  # TODO: keep for debug
    my_videos = all_videos[args.job_id :: args.num_jobs]

    if not my_videos:
        print(f"Job {args.job_id}: no videos assigned.", flush=True)
        return

    my_df = df.filter(pl.col("video_uid").is_in(my_videos))
    n_rows = len(my_df)
    print(f"Job {args.job_id}: {len(my_videos)} videos, {n_rows} rows", flush=True)

    # Pre-group rows by video to avoid repeated filtering inside threads
    grouped = {uid: my_df.filter(pl.col("video_uid") == uid).to_dicts() for uid in my_videos}

    total_done = total_skipped = total_errors = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_video, uid, rows, OUT_DIR, vmeta): uid for uid, rows in grouped.items()}
        for future in as_completed(futures):
            uid = futures[future]
            try:
                done, skipped, errors = future.result()
                total_done += done
                total_skipped += skipped
                total_errors += errors
                print(
                    f"  {uid}: {done} written, {skipped} skipped, {errors} errors",
                    flush=True,
                )
            except Exception as exc:
                print(f"  [CRASH] {uid}: {exc}", flush=True)
                total_errors += 1

    print(
        f"Job {args.job_id} finished — {total_done} written, {total_skipped} skipped, {total_errors} errors",
        flush=True,
    )


if __name__ == "__main__":
    main()
