#!/usr/bin/env python3
"""Does timed_caption-guided trimming improve Parakeet WER on LN?

Re-runs Parakeet on the SAME 24 samples used in verify_caption, comparing:
  A) full clip (raw OGG → 16 kHz mono WAV)
  B) trimmed clip (audio[min(start_time) : max(end_time) + pad])

Reports per-condition micro-WER vs LN caption.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

PARAKEET = (
    "/capstor/store/cscs/swissai/infra01/MLLM/audio_asr/"
    "parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo"
)

GROUPS = {
    "coco_val":          "coco_val_localized_narratives.jsonl",
    "ade20k_validation": "ade20k_validation_localized_narratives.jsonl",
    "flickr30k_test":    "flickr30k_test_localized_narratives.jsonl",
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(r, h):
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): d[i][0] = i
    for j in range(len(h) + 1): d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    return d[-1][-1]


def sample_records(source_root: Path, group: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    files = sorted((source_root / "annotations").glob(GROUPS[group]))
    chosen: list[dict] = []
    for path in files:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                rec = json.loads(line)
                if len(chosen) < n:
                    chosen.append(rec)
                else:
                    if rng.random() < n / (len(chosen) + 1):
                        chosen[rng.randint(0, len(chosen) - 1)] = rec
    return chosen[:n]


def decode_to_wav(ogg_path: Path, wav_path: Path,
                  start_s: float | None = None,
                  end_s: float | None = None,
                  target_sr: int = 16000) -> float:
    """Decode OGG → 16 kHz mono WAV, optionally trimmed. Returns clip duration in s."""
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    audio, sr = sf.read(str(ogg_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    if start_s is not None and end_s is not None:
        s = max(0, int(start_s * sr))
        e = min(len(audio), int(end_s * sr))
        if e > s:
            audio = audio[s:e]
    sf.write(str(wav_path), audio, sr, subtype="PCM_16")
    return len(audio) / sr


def hyp_text(o):
    if hasattr(o, "text"): return o.text
    if isinstance(o, list) and o and hasattr(o[0], "text"): return o[0].text
    return str(o)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path,
                   default=Path("/capstor/store/cscs/swissai/infra01/audio-datasets/raw/localized_narratives"))
    p.add_argument("--n-per-group", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pad-ms", type=int, default=100,
                   help="Padding around timed_caption span (ms each side)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--model-path", default=PARAKEET)
    args = p.parse_args()

    audio_root = args.source_root / "voice-recordings"
    pad_s = args.pad_ms / 1000.0

    # Sample
    samples: list[dict] = []
    for g in GROUPS:
        for r in sample_records(args.source_root, g, args.n_per_group, args.seed):
            r["_group"] = g
            samples.append(r)
    print(f"Sampled {len(samples)} records (seed={args.seed})", flush=True)

    # Decode both conditions
    tmpdir = Path(tempfile.mkdtemp(prefix="ln_trim_"))
    full_wavs, trim_wavs = [], []
    full_durs, trim_durs = [], []
    for i, rec in enumerate(samples):
        ogg = audio_root / rec["voice_recording"]
        tc = rec["timed_caption"]
        start = max(0.0, min(t["start_time"] for t in tc) - pad_s)
        end = max(t["end_time"] for t in tc) + pad_s

        full_w = tmpdir / f"{i:03d}_full.wav"
        trim_w = tmpdir / f"{i:03d}_trim.wav"
        full_d = decode_to_wav(ogg, full_w, None, None)
        trim_d = decode_to_wav(ogg, trim_w, start, end)
        full_wavs.append(full_w); trim_wavs.append(trim_w)
        full_durs.append(full_d); trim_durs.append(trim_d)

    print(f"Avg full clip duration:    {np.mean(full_durs):.2f}s", flush=True)
    print(f"Avg trimmed duration:      {np.mean(trim_durs):.2f}s", flush=True)
    print(f"Avg trimmed-off silence:   {np.mean(np.array(full_durs)-np.array(trim_durs)):.2f}s", flush=True)

    # Load Parakeet ONCE, run twice
    print(f"Loading Parakeet from {args.model_path}", flush=True)
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.restore_from(args.model_path, map_location=args.device)
    model.eval()
    if hasattr(model, "to"): model = model.to(args.device)

    print("Transcribing FULL clips ...", flush=True)
    out_full = model.transcribe([str(p) for p in full_wavs], batch_size=args.batch_size)
    print("Transcribing TRIMMED clips ...", flush=True)
    out_trim = model.transcribe([str(p) for p in trim_wavs], batch_size=args.batch_size)

    asr_full = [hyp_text(o) for o in out_full]
    asr_trim = [hyp_text(o) for o in out_trim]

    # Score
    def score(asr_list):
        tw, tw_ref, tc_, tc_ref = 0, 0, 0, 0
        rows = []
        for rec, asr in zip(samples, asr_list):
            ref = normalize(rec["caption"])
            hyp = normalize(asr)
            r_w, h_w = ref.split(), hyp.split()
            we = edit_distance(r_w, h_w)
            ce = edit_distance(list(ref), list(hyp))
            tw += we; tw_ref += len(r_w); tc_ += ce; tc_ref += len(ref)
            rows.append({"group": rec["_group"], "image_id": rec["image_id"],
                         "wer": we/max(len(r_w),1), "cer": ce/max(len(ref),1),
                         "asr": asr, "ref": rec["caption"]})
        return rows, (tw/tw_ref, tc_/tc_ref, tw, tw_ref, tc_, tc_ref)

    rows_full, (wer_full, cer_full, twf, twrf, tcf, tcrf) = score(asr_full)
    rows_trim, (wer_trim, cer_trim, twt, twrt, tct, tcrt) = score(asr_trim)

    print("\n" + "=" * 72)
    print("Result")
    print("=" * 72)
    print(f"  FULL    micro-WER: {wer_full:.4f}  ({twf}/{twrf})  micro-CER: {cer_full:.4f}")
    print(f"  TRIMMED micro-WER: {wer_trim:.4f}  ({twt}/{twrt})  micro-CER: {cer_trim:.4f}")
    print(f"  Δ WER (trim-full): {wer_trim - wer_full:+.4f}")
    print(f"  Δ CER (trim-full): {cer_trim - cer_full:+.4f}")

    # Per-sample diff: where does trimming move WER?
    print("\nPer-sample WER diff (sorted by impact):")
    diffs = []
    for f, t in zip(rows_full, rows_trim):
        diffs.append((f["wer"], t["wer"], t["wer"] - f["wer"], f, t))
    diffs.sort(key=lambda x: x[2])
    print(f"{'group':<20} {'image_id':<14} {'full WER':>9} {'trim WER':>9} {'Δ':>9}")
    for fwer, twer, d, f, t in diffs[:5] + diffs[-5:]:
        print(f"{f['group']:<20} {f['image_id']:<14} {fwer:>9.3f} {twer:>9.3f} {d:>+9.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
