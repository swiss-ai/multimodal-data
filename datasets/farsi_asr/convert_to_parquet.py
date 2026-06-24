"""
Convert farsi-asr/farsi-asr-dataset (tar.gz of wav+txt pairs) to HF-style parquet.

Each tar.gz contains paired {stem}.wav + {stem}.txt files.
Each worker processes ONE tar.gz in a single streaming pass and writes its OWN
parquet file — no audio bytes ever cross process boundaries.

Usage:
    python convert_to_parquet.py \
        --input-dir /path/to/farsi-asr-dataset \
        --output-dir /path/to/farsi-asr-dataset/parquet \
        --num-workers 288
"""
import argparse
import tarfile
from multiprocessing import Pool
from pathlib import Path

import polars as pl


_FLUSH_BYTES = 200 * 1024 * 1024  # flush parquet every ~200MB of audio
_BUFFER_MAX_BYTES = 200 * 1024 * 1024  # cap unpaired buffer at ~200MB; drop oldest


def _write_part(rows, output_dir, tar_name, part_idx):
    out_path = Path(output_dir) / f"{tar_name}_part{part_idx:04d}.parquet"
    pl.DataFrame(rows).write_parquet(out_path)
    return out_path.name


def process_one_tar(args):
    """Read tar.gz (streaming single-pass), pair wav+txt by stem, flush incrementally.

    Memory is bounded in two ways:
      - Unpaired buffer is capped at _BUFFER_MAX_BYTES (drops oldest if exceeded).
      - Completed rows are flushed to disk every _FLUSH_BYTES of audio bytes.
    """
    tar_path, output_dir = args
    tar_path = Path(tar_path)
    tar_name = tar_path.name.replace(".tar.gz", "")
    output_dir = Path(output_dir)

    # Resume: skip if any part exists for this tar
    if any(output_dir.glob(f"{tar_name}_part*.parquet")):
        return tar_name, -1, "skipped"

    rows = []
    rows_bytes = 0
    buffer = {}          # stem -> {".wav": bytes, ".txt": bytes}
    buffer_bytes = 0
    buffer_order = []    # FIFO for eviction
    total_samples = 0
    part_idx = 0

    try:
        with tarfile.open(tar_path, "r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                p = Path(m.name)
                ext = p.suffix.lower()
                if ext not in (".wav", ".txt"):
                    continue
                stem = p.stem
                try:
                    data = tf.extractfile(m).read()
                except Exception:
                    continue

                slot = buffer.get(stem)
                if slot is None:
                    slot = {}
                    buffer[stem] = slot
                    buffer_order.append(stem)
                slot[ext] = data
                buffer_bytes += len(data)

                if ".wav" in slot and ".txt" in slot:
                    transcription = slot[".txt"].decode("utf-8", errors="ignore").strip()
                    buffer_bytes -= (len(slot[".wav"]) + len(slot[".txt"]))
                    wav_bytes = slot[".wav"]
                    del buffer[stem]
                    buffer_order.remove(stem)
                    if transcription:
                        rows.append({
                            "audio": {"bytes": wav_bytes, "path": f"{tar_name}/{stem}.wav"},
                            "transcription": transcription,
                            "source": tar_name,
                        })
                        rows_bytes += len(wav_bytes) + len(transcription.encode("utf-8"))

                    # Flush completed rows if buffer limit hit
                    if rows_bytes >= _FLUSH_BYTES:
                        _write_part(rows, output_dir, tar_name, part_idx)
                        total_samples += len(rows)
                        part_idx += 1
                        rows = []
                        rows_bytes = 0

                # Evict oldest unpaired entries if buffer too large
                while buffer_bytes > _BUFFER_MAX_BYTES and buffer_order:
                    old = buffer_order.pop(0)
                    if old in buffer:
                        old_slot = buffer.pop(old)
                        buffer_bytes -= sum(len(v) for v in old_slot.values())
    except Exception as e:
        # Save whatever we have before the error
        if rows:
            _write_part(rows, output_dir, tar_name, part_idx)
            total_samples += len(rows)
        return tar_name, total_samples, f"error: {e}"

    if rows:
        _write_part(rows, output_dir, tar_name, part_idx)
        total_samples += len(rows)
    return tar_name, total_samples, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=288)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all tar.gz files from radio/ and youtube/
    tasks = []
    for sub in ["radio", "youtube"]:
        sub_dir = input_dir / sub
        if not sub_dir.exists():
            continue
        for tar_path in sorted(sub_dir.glob("*.tar.gz")):
            tasks.append((str(tar_path), str(output_dir)))

    num_workers = min(args.num_workers, len(tasks))
    print(f"Total tar.gz: {len(tasks)}, workers: {num_workers}")

    total_samples = 0
    completed = 0
    with Pool(num_workers) as pool:
        # imap_unordered transfers only small tuples (name, count, status) back
        for tar_name, n, status in pool.imap_unordered(process_one_tar, tasks):
            completed += 1
            if status == "ok":
                total_samples += n
                print(f"[{completed}/{len(tasks)}] {tar_name}.parquet ({n} samples)")
            elif status == "skipped":
                print(f"[{completed}/{len(tasks)}] {tar_name}.parquet SKIPPED (exists)")
            else:
                print(f"[{completed}/{len(tasks)}] {tar_name} FAILED: {status}")

    print(f"Done. {total_samples} samples across {completed} parquet files.")


if __name__ == "__main__":
    main()
