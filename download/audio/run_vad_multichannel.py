#!/usr/bin/env python3
"""Run Silero VAD on WDS tars with optional per-channel splitting.

For multi-channel (stereo/full-duplex) datasets where each channel is a
separate speaker, splits channels and runs VAD independently on each,
producing entries keyed as {stem}_ch{N}.

For mono audio or without --split-channels, behaves like standard VAD.

Output JSONL:
    {"stem_ch0": {"timestamps": [[s,e],...], "duration_sec": float, "sample_rate": 16000, "channel": 0}}
    Timestamps are in sample indices at 16 kHz.

Usage:
    python run_vad_multichannel.py \
        --wds-dir /capstor/.../data/train \
        --output /capstor/.../vad_results.jsonl \
        --num-workers 61 --split-channels
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import io
import logging
import multiprocessing as mp
import tarfile
from pathlib import Path

import orjson

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VAD_SAMPLE_RATE = 16000
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".opus", ".ogg")

_worker_vad = None


def _init_worker(onnx: bool):
    global _worker_vad
    logging.disable(logging.INFO)
    from silero_vad import load_silero_vad
    _worker_vad = load_silero_vad(onnx=onnx)


def _read_done_keys(path: Path) -> set:
    """Read already-processed keys from a JSONL file for resume."""
    done = set()
    if path.is_file():
        with open(path, "rb") as f:
            for line in f:
                try:
                    done.add(next(iter(orjson.loads(line))))
                except Exception:
                    pass
    return done


def _process_tar(args_tuple) -> list[bytes]:
    tar_path, split_channels, done_keys = args_tuple
    import soundfile as sf
    import torch
    import torchaudio
    from silero_vad import get_speech_timestamps

    results = []
    try:
        tf = tarfile.open(tar_path)
    except (EOFError, tarfile.ReadError, OSError) as e:
        logger.warning(f"Skipping corrupt tar {tar_path}: {e}")
        return results

    with tf:
        # First pass: read JSON sidecars for speaker metadata
        speaker_maps = {}  # stem -> {channel: speaker_id}
        for member in tf:
            if member.isfile() and member.name.endswith(".json"):
                stem = member.name[:member.name.rfind(".")]
                try:
                    meta = orjson.loads(tf.extractfile(member).read())
                    if isinstance(meta, dict) and "speakers" in meta:
                        speaker_maps[stem] = {
                            s["channel"]: s.get("speaker_id", f"spk_{s['channel']}")
                            for s in meta["speakers"]
                        }
                except Exception:
                    pass

        # Rewind tar for second pass: audio members
        tf.members = []
        tf.offset = 0
        try:
            tf = tarfile.open(tar_path)
        except Exception:
            return results

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
                logger.warning(f"Failed to read {stem}: {e}")
                continue

            duration_sec = data.shape[0] / sr
            num_ch = data.shape[1] if data.ndim > 1 else 1

            if split_channels and num_ch > 1:
                channels = [(data[:, ch], ch) for ch in range(num_ch)]
            else:
                mono = data.mean(axis=1) if data.ndim > 1 else data
                channels = [(mono, None)]

            for ch_data, ch_idx in channels:
                if sr != VAD_SAMPLE_RATE:
                    wav = torchaudio.functional.resample(
                        torch.from_numpy(ch_data).unsqueeze(0), sr, VAD_SAMPLE_RATE,
                    ).squeeze(0)
                else:
                    wav = torch.from_numpy(ch_data)

                try:
                    ts = get_speech_timestamps(wav, _worker_vad, sampling_rate=VAD_SAMPLE_RATE)
                except Exception as e:
                    logger.warning(f"VAD failed on {stem} ch{ch_idx}: {e}")
                    continue

                key = f"{stem}_ch{ch_idx}" if ch_idx is not None else stem
                if key in done_keys:
                    continue
                entry = {
                    "timestamps": [[t["start"], t["end"]] for t in ts],
                    "duration_sec": duration_sec,
                    "sample_rate": VAD_SAMPLE_RATE,
                }
                if ch_idx is not None:
                    entry["channel"] = ch_idx
                    spk_map = speaker_maps.get(stem, {})
                    if ch_idx in spk_map:
                        entry["speaker_id"] = spk_map[ch_idx]
                results.append(orjson.dumps({key: entry}))

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Silero VAD on WDS tars (multi-channel aware)")
    parser.add_argument("--wds-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--split-channels", action="store_true",
                        help="Run VAD per channel instead of downmixing to mono")
    parser.add_argument("--backend", choices=["onnx", "jit"], default="onnx")
    parser.add_argument("--pattern", default="*.tar")
    args = parser.parse_args()

    tar_files = sorted(args.wds_dir.glob(args.pattern))
    if not tar_files:
        raise FileNotFoundError(f"No tars matching {args.pattern} in {args.wds_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_keys = _read_done_keys(args.output)
    if done_keys:
        logger.info(f"Resuming: {len(done_keys):,} entries already done")

    logger.info(f"{len(tar_files)} tars, split_channels={args.split_channels}")
    work = [(str(t), args.split_channels, done_keys) for t in tar_files]

    n_workers = min(args.num_workers, len(tar_files))
    ctx = mp.get_context("spawn")
    processed = 0
    from tqdm import tqdm
    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(args.backend == "onnx",)) as pool:
        with open(args.output, "ab") as out:
            for results in tqdm(pool.imap_unordered(_process_tar, work), total=len(work), desc="VAD"):
                for line in results:
                    out.write(line + b"\n")
                    processed += 1
                out.flush()

    logger.info(f"Done: {processed:,} new entries -> {args.output}")


if __name__ == "__main__":
    main()
