#!/usr/bin/env python3
"""Convert MRSAudio music/sing to merged audio-only SHAR.

Concatenates consecutive segments within each folder into ≤30s chunks,
writes directly to SharWriter.

Usage:
    python prepare_music_sing_to_shar.py \
        --data-root /capstor/.../hf___MRSAudio___MRSAudio_git_clone \
        --split MRSMusic \
        --shar-dir /capstor/.../SHAR_TODO/annotate/mrsaudio_music

    python prepare_music_sing_to_shar.py \
        --data-root /capstor/.../hf___MRSAudio___MRSAudio_git_clone \
        --split MRSSing \
        --shar-dir /capstor/.../SHAR_TODO/annotate/mrsaudio_sing
"""

import argparse
import io
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_CHUNK_DUR = 30.0


def _scan_folder(args):
    import soundfile as sf
    mf_path, data_root, split = args
    mf = Path(mf_path)
    folder = mf.parent.name
    data = json.loads(mf.read_text())
    entries = list(data.values()) if isinstance(data, dict) else data

    segments = []
    for entry in entries:
        wav_fn = entry.get("mono_wav_fn") or entry.get("wav_fn")
        if not wav_fn:
            continue
        wav_path = Path(data_root) / wav_fn
        if not wav_path.is_file():
            continue
        try:
            dur = sf.info(str(wav_path)).duration
        except Exception:
            continue
        segments.append((str(wav_path), dur))

    chunks = []
    cur, cur_dur, idx = [], 0.0, 0
    for wp, dur in segments:
        if cur_dur + dur > MAX_CHUNK_DUR and cur:
            chunks.append((folder, idx, cur))
            cur, cur_dur, idx = [], 0.0, idx + 1
        cur.append(wp)
        cur_dur += dur
    if cur:
        chunks.append((folder, idx, cur))
    return chunks


def _worker(args_tuple):
    import numpy as np
    import soundfile as sf
    from lhotse import Recording
    from lhotse.shar import SharWriter

    worker_id, chunks, shar_dir, target_sr, shard_size, shar_format, split = args_tuple
    worker_dir = Path(shar_dir) / f"worker_{worker_id:02d}"
    if (worker_dir / "_SUCCESS").is_file():
        return {"worker_id": worker_id, "written": -1}
    if worker_dir.is_dir():
        import shutil
        shutil.rmtree(worker_dir)
    worker_dir.mkdir(parents=True)

    t0 = time.time()
    written = skipped = 0
    total_dur = 0.0

    with SharWriter(str(worker_dir), fields={"recording": shar_format}, shard_size=shard_size) as writer:
        for folder, chunk_idx, wav_paths in chunks:
            arrays, sr = [], None
            for wp in wav_paths:
                try:
                    data, file_sr = sf.read(wp, dtype="float32")
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    if sr is None:
                        sr = file_sr
                    arrays.append(data)
                except Exception:
                    continue
            if not arrays:
                continue

            merged = np.concatenate(arrays)
            if sr != target_sr:
                import torchaudio, torch
                merged = torchaudio.functional.resample(
                    torch.from_numpy(merged).unsqueeze(0), sr, target_sr
                ).squeeze(0).numpy()

            rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(merged ** 2))) + 1e-10)
            if rms_db < -40.0:
                skipped += 1
                continue

            source_id = f"{split}/{folder}"
            rec_id = f"{source_id}@{chunk_idx:06d}"
            buf = io.BytesIO()
            sf.write(buf, merged, target_sr, format="WAV", subtype="FLOAT")
            buf.seek(0)
            rec = Recording.from_bytes(buf.read(), recording_id=rec_id)
            cut = rec.to_cut()
            cut.custom = {"folder": folder, "split": split, "num_segments": len(wav_paths), "rms_db": rms_db, "clip_start": 0.0}
            writer.write(cut)
            written += 1
            total_dur += cut.duration

    logger.info(f"Worker {worker_id}: {written} written, {skipped} skipped in {time.time()-t0:.1f}s")
    (worker_dir / "_SUCCESS").write_text("ok\n")
    return {"worker_id": worker_id, "written": written, "skipped": skipped, "total_duration_sec": total_dur}


def main():
    parser = argparse.ArgumentParser(description="MRSAudio music/sing → merged audio-only SHAR")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True, help="MRSMusic or MRSSing")
    parser.add_argument("--shar-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--shar-format", type=str, default="flac")
    args = parser.parse_args()

    metadata_files = sorted((args.data_root / args.split).glob("*/metadata.json"))
    logger.info(f"{len(metadata_files)} folders in {args.split}")

    scan_work = [(str(mf), str(args.data_root), args.split) for mf in metadata_files]
    all_chunks = []
    with mp.Pool(min(args.num_workers, len(scan_work))) as pool:
        for fc in pool.imap_unordered(_scan_folder, scan_work, chunksize=4):
            all_chunks.extend(fc)
    logger.info(f"{len(all_chunks)} merged chunks")

    args.shar_dir.mkdir(parents=True, exist_ok=True)
    n = min(args.num_workers, len(all_chunks))
    buckets = [[] for _ in range(n)]
    for i, c in enumerate(all_chunks):
        buckets[i % n].append(c)

    work = [(wid, b, str(args.shar_dir), args.target_sr, args.shard_size, args.shar_format, args.split)
            for wid, b in enumerate(buckets) if b]

    ctx = mp.get_context("forkserver")
    with ctx.Pool(len(work)) as pool:
        results = pool.map(_worker, work)

    total = sum(r["written"] for r in results if r["written"] >= 0)
    hours = sum(r.get("total_duration_sec", 0) for r in results) / 3600
    logger.info(f"Done: {total} cuts, {hours:.1f}h")


if __name__ == "__main__":
    main()
