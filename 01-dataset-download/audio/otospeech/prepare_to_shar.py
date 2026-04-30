#!/usr/bin/env python3
"""Convert otoSpeech stereo WDS tars to mono per-speaker SHAR using VAD.

Reads stereo FLACs from WDS tars, splits into per-channel mono,
applies per-channel VAD segmentation with merging, and writes to SHAR
with speaker_id and channel metadata.

Usage:
    python prepare_to_shar.py \
        --wds-dir /capstor/.../data/train \
        --vad-dir /capstor/.../vad_per_shard \
        --shar-dir /capstor/.../SHAR/stage_1/otospeech \
        --num-workers 61
"""

import argparse
import io
import json
import logging
import multiprocessing as mp
import tarfile
import time
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".opus", ".ogg")
VAD_SAMPLE_RATE = 16000


def _load_vad_and_speakers(vad_file: Path) -> tuple[dict, dict]:
    """Load per-channel VAD entries and speaker IDs from a per-shard JSONL."""
    vad = {}
    speakers = {}
    with open(vad_file) as f:
        for line in f:
            d = json.loads(line)
            key, val = next(iter(d.items()))
            vad[key] = val["timestamps"]
            if "speaker_id" in val:
                speakers[key] = val["speaker_id"]
    return vad, speakers


def _merge_timestamps(timestamps, merge_gap_sec):
    """Merge adjacent VAD segments within merge_gap_sec. Returns [(start_sec, dur_sec), ...]."""
    if not timestamps:
        return []
    sr = float(VAD_SAMPLE_RATE)
    spans = [(s / sr, e / sr) for s, e in timestamps if e > s]
    if not spans:
        return []

    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start - merged[-1][1] <= merge_gap_sec:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [(s, e - s) for s, e in merged]


def _process_tar(args_tuple):
    """Process one tar: split stereo, apply VAD, write SHAR."""
    (tar_path, vad_dir, shar_dir, worker_id,
     target_sr, shard_size, merge_gap_sec, shar_format) = args_tuple

    import numpy as np
    import soundfile as sf
    import torchaudio
    import torch
    from lhotse import Recording, MonoCut
    from lhotse.shar import SharWriter

    worker_dir = Path(shar_dir) / f"worker_{worker_id:02d}"
    success_marker = worker_dir / "_SUCCESS"
    if success_marker.is_file():
        logger.info(f"Worker {worker_id}: reusing completed output")
        return {"worker_id": worker_id, "written": -1, "reused": True}

    if worker_dir.is_dir():
        import shutil
        shutil.rmtree(worker_dir)
    worker_dir.mkdir(parents=True)

    # Load VAD for this shard
    shard_name = f"train_{Path(tar_path).stem}"
    vad_file = Path(vad_dir) / f"{shard_name}.jsonl"
    if not vad_file.is_file():
        logger.warning(f"Worker {worker_id}: no VAD file {vad_file}")
        success_marker.write_text("ok\n")
        return {"worker_id": worker_id, "written": 0}

    vad, speakers = _load_vad_and_speakers(vad_file)

    # Load speaker metadata from JSON sidecars
    sidecar_speakers = {}
    try:
        tf = tarfile.open(tar_path)
    except Exception as e:
        logger.warning(f"Worker {worker_id}: corrupt tar {tar_path}: {e}")
        success_marker.write_text("ok\n")
        return {"worker_id": worker_id, "written": 0}

    t0 = time.time()
    written = skipped = 0
    total_dur = 0.0
    stats = Counter()

    with tf, SharWriter(str(worker_dir), fields={"recording": shar_format}, shard_size=shard_size) as writer:
        # First pass: read JSON sidecars for speaker info
        for member in tf:
            if member.isfile() and member.name.endswith(".json"):
                stem = member.name[:-5]
                try:
                    meta = json.loads(tf.extractfile(member).read())
                    if isinstance(meta, dict) and "speakers" in meta:
                        sidecar_speakers[stem] = {
                            s["channel"]: s.get("speaker_id", f"spk_{s['channel']}")
                            for s in meta["speakers"]
                        }
                except Exception:
                    pass

        # Reopen for audio pass
        tf.close()
        tf = tarfile.open(tar_path)

        for member in tf:
            if not member.isfile():
                continue
            ext = member.name[member.name.rfind("."):]
            if ext not in AUDIO_SUFFIXES:
                continue

            stem = member.name[:member.name.rfind(".")]
            try:
                data, sr = sf.read(io.BytesIO(tf.extractfile(member).read()), dtype="float32")
            except Exception as e:
                logger.warning(f"Worker {worker_id}: failed to read {stem}: {e}")
                stats["read_failed"] += 1
                continue

            num_ch = data.shape[1] if data.ndim > 1 else 1

            for ch in range(num_ch):
                ch_key = f"{stem}_ch{ch}"
                ch_timestamps = vad.get(ch_key)
                if ch_timestamps is None:
                    stats["no_vad"] += 1
                    continue

                # Extract mono channel
                ch_data = data[:, ch] if data.ndim > 1 else data

                # Resample
                if sr != target_sr:
                    ch_tensor = torchaudio.functional.resample(
                        torch.from_numpy(ch_data).unsqueeze(0), sr, target_sr
                    ).squeeze(0).numpy()
                else:
                    ch_tensor = ch_data

                # Merge VAD segments
                chunk_count = 0
                chunks = _merge_timestamps(ch_timestamps, merge_gap_sec)
                if not chunks:
                    stats["empty_vad"] += 1
                    continue

                # Get speaker ID
                spk_id = speakers.get(ch_key)
                if not spk_id:
                    spk_map = sidecar_speakers.get(stem, {})
                    spk_id = spk_map.get(ch, f"spk_{ch}")

                for offset_sec, dur_sec in chunks:
                    s_idx = int(offset_sec * target_sr)
                    e_idx = int((offset_sec + dur_sec) * target_sr)
                    e_idx = min(e_idx, len(ch_tensor))
                    segment = ch_tensor[s_idx:e_idx]

                    if len(segment) == 0:
                        stats["empty_segment"] += 1
                        continue

                    # Compute RMS
                    rms = float(np.sqrt(np.mean(segment ** 2)))
                    rms_db = 20.0 * np.log10(rms + 1e-10)
                    if rms_db < -40.0:
                        stats["skipped_quiet"] += 1
                        skipped += 1
                        continue

                    # Write segment to in-memory WAV, then create Recording
                    buf = io.BytesIO()
                    sf.write(buf, segment, target_sr, format="WAV", subtype="FLOAT")
                    buf.seek(0)
                    source_id = f"{stem}_ch{ch}"
                    rec = Recording.from_bytes(buf.read(), recording_id=f"{source_id}@{chunk_count:06d}")
                    cut = rec.to_cut()
                    cut.custom = {
                        "source_recording_id": stem,
                        "clip_start": offset_sec,
                        "global_offset_sec": offset_sec,
                        "channel": ch,
                        "speaker_id": spk_id,
                        "rms_db": rms_db,
                    }
                    chunk_count += 1

                    writer.write(cut)
                    written += 1
                    total_dur += cut.duration

        tf.close()

    elapsed = time.time() - t0
    logger.info(f"Worker {worker_id}: {written} written, {skipped} skipped in {elapsed:.1f}s ({stats})")
    success_marker.write_text("ok\n")

    result = {
        "worker_id": worker_id,
        "written": written,
        "skipped": skipped,
        "total_duration_sec": total_dur,
        "elapsed_sec": elapsed,
        "stats": dict(stats),
    }
    (worker_dir / "worker_stats.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="otoSpeech stereo WDS → per-speaker mono SHAR")
    parser.add_argument("--wds-dir", type=Path, required=True)
    parser.add_argument("--vad-dir", type=Path, required=True)
    parser.add_argument("--shar-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=61)
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--merge-gap-sec", type=float, default=0.5)
    parser.add_argument("--shar-format", type=str, default="flac")
    args = parser.parse_args()

    tar_files = sorted(args.wds_dir.glob("*.tar"))
    if not tar_files:
        raise FileNotFoundError(f"No tars in {args.wds_dir}")

    args.shar_dir.mkdir(parents=True, exist_ok=True)

    work = [
        (str(t), str(args.vad_dir), str(args.shar_dir), i,
         args.target_sr, args.shard_size, args.merge_gap_sec, args.shar_format)
        for i, t in enumerate(tar_files)
    ]

    n_workers = min(args.num_workers, len(tar_files))
    logger.info(f"{len(tar_files)} tars, {n_workers} workers, merge_gap={args.merge_gap_sec}s")

    ctx = mp.get_context("forkserver")
    with ctx.Pool(n_workers) as pool:
        results = pool.map(_process_tar, work)

    total_written = sum(r["written"] for r in results if r["written"] >= 0)
    total_dur = sum(r.get("total_duration_sec", 0) for r in results)
    logger.info(f"Done: {total_written} cuts, {total_dur/3600:.1f}h")


if __name__ == "__main__":
    main()
