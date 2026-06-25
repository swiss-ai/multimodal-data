#!/usr/bin/env python3
"""Verify whether LN's `caption` is verbatim-from-audio or rephrased.

For N sampled records, compare:
  A) `caption`                               — full transcript field
  B) joined-words from `timed_caption`       — the per-word timestamps reassembled
  C) Parakeet ASR run on the OGG audio       — independent transcription

Hypothesis check:
- If A == B (after normalization), `caption` IS the timed-caption concatenation.
- If C ≈ A, the captions are essentially verbatim transcripts of the audio
  (ASR-level disagreement only).
- If C diverges from A in systematic ways (deleted hesitations, normalized
  numbers, removed restarts), the captions are cleaned/rephrased.

Usage:
  /opt/venv/bin/python verify_caption.py \
      --groups coco_val,ade20k_validation,flickr30k_test \
      --n-per-group 8 \
      --device cuda

Requires: NeMo + soundfile + scipy. Run on a GPU node.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

PARAKEET = (
    "/capstor/store/cscs/swissai/infra01/MLLM/audio_asr/"
    "parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo"
)

GROUPS = {
    "open_images_train":      "open_images_train_v6_localized_narratives*.jsonl",
    "open_images_validation": "open_images_validation_localized_narratives.jsonl",
    "open_images_test":       "open_images_test_localized_narratives.jsonl",
    "coco_train":             "coco_train_localized_narratives*.jsonl",
    "coco_val":               "coco_val_localized_narratives.jsonl",
    "flickr30k_train":        "flickr30k_train_localized_narratives.jsonl",
    "flickr30k_val":          "flickr30k_val_localized_narratives.jsonl",
    "flickr30k_test":         "flickr30k_test_localized_narratives.jsonl",
    "ade20k_train":           "ade20k_train_localized_narratives.jsonl",
    "ade20k_validation":      "ade20k_validation_localized_narratives.jsonl",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def wer(ref: str, hyp: str) -> float:
    """Word error rate on already-normalized strings."""
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    # classic DP
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / len(r)


def join_timed(timed: list[dict]) -> str:
    return " ".join(t["utterance"] for t in timed if t.get("utterance"))


def sample_records(source_root: Path, group: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    files = sorted((source_root / "annotations").glob(GROUPS[group]))
    if not files:
        raise FileNotFoundError(f"no annotations for group {group}")
    # Reservoir sample across all shards.
    chosen: list[dict] = []
    for path in files:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if len(chosen) < n:
                    chosen.append(rec)
                else:
                    j = rng.randint(0, len(chosen) - 1)
                    if rng.random() < n / (len(chosen) + 1):
                        chosen[j] = rec
    return chosen[:n]


def decode_ogg_to_wav(ogg_path: Path, wav_path: Path, target_sr: int = 16000) -> None:
    """Decode Ogg/Vorbis → 16 kHz mono WAV using soundfile + scipy resample.

    libsndfile (under soundfile) handles Ogg/Vorbis natively. Parakeet
    expects 16 kHz mono; LN audio is 48 kHz stereo, so we downmix and
    resample.
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(str(ogg_path), dtype="float32", always_2d=True)
    # downmix to mono
    audio = audio.mean(axis=1)
    if sr != target_sr:
        # resample_poly is faster than scipy.signal.resample for integer ratios
        from math import gcd
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    sf.write(str(wav_path), audio, target_sr, subtype="PCM_16")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-root", type=Path,
                   default=Path("/capstor/store/cscs/swissai/infra01/audio-datasets/raw/localized_narratives"))
    p.add_argument("--groups", default="coco_val,ade20k_validation,flickr30k_test",
                   help="comma-separated LN group names")
    p.add_argument("--n-per-group", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--model-path", default=PARAKEET)
    p.add_argument("--output-json", type=Path, default=None,
                   help="optional path to dump per-sample results")
    args = p.parse_args()

    group_names = [g.strip() for g in args.groups.split(",") if g.strip()]
    audio_root = args.source_root / "voice-recordings"

    # 1. Sample records
    print(f"Sampling {args.n_per_group} per group from {len(group_names)} groups (seed={args.seed})", flush=True)
    samples: list[dict] = []
    for g in group_names:
        recs = sample_records(args.source_root, g, args.n_per_group, args.seed)
        for r in recs:
            r["_group"] = g
        samples.extend(recs)
    print(f"  {len(samples)} samples total", flush=True)

    # 2. Convert OGG → WAV in tempdir
    tmpdir = Path(tempfile.mkdtemp(prefix="ln_verify_"))
    print(f"Decoding OGG → 16 kHz mono WAV in {tmpdir}", flush=True)
    wav_paths: list[Path] = []
    for i, rec in enumerate(samples):
        ogg = audio_root / rec["voice_recording"]
        wav = tmpdir / f"{i:04d}.wav"
        decode_ogg_to_wav(ogg, wav)
        wav_paths.append(wav)
    print(f"  decoded {len(wav_paths)} clips", flush=True)

    # 3. Load Parakeet and transcribe
    print(f"Loading Parakeet from {args.model_path}", flush=True)
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.restore_from(args.model_path, map_location=args.device)
    model.eval()
    if hasattr(model, "to"):
        model = model.to(args.device)
    print("Transcribing ...", flush=True)
    raw_outputs = model.transcribe([str(p) for p in wav_paths], batch_size=args.batch_size)

    def hyp_text(o) -> str:
        # NeMo returns either Hypothesis or str depending on version
        if hasattr(o, "text"):
            return o.text
        if isinstance(o, list) and o and hasattr(o[0], "text"):
            return o[0].text
        return str(o)

    asr_texts = [hyp_text(o) for o in raw_outputs]

    # 4. Compare
    print("\n" + "=" * 80)
    print("Per-sample comparison")
    print("=" * 80)
    results = []
    agg = {"wer_caption_vs_asr": [], "wer_timed_vs_asr": [],
           "wer_caption_vs_timed": [], "char_ratio_caption_vs_timed": []}

    for rec, asr in zip(samples, asr_texts):
        cap_norm = normalize_text(rec["caption"])
        timed_norm = normalize_text(join_timed(rec["timed_caption"]))
        asr_norm = normalize_text(asr)

        w_cap_asr = wer(cap_norm, asr_norm)
        w_tim_asr = wer(timed_norm, asr_norm)
        w_cap_tim = wer(cap_norm, timed_norm)
        cr_cap_tim = SequenceMatcher(None, cap_norm, timed_norm).ratio()

        agg["wer_caption_vs_asr"].append(w_cap_asr)
        agg["wer_timed_vs_asr"].append(w_tim_asr)
        agg["wer_caption_vs_timed"].append(w_cap_tim)
        agg["char_ratio_caption_vs_timed"].append(cr_cap_tim)

        results.append({
            "group": rec["_group"],
            "image_id": rec["image_id"],
            "annotator_id": rec["annotator_id"],
            "voice_recording": rec["voice_recording"],
            "caption": rec["caption"],
            "timed_caption_joined": join_timed(rec["timed_caption"]),
            "asr": asr,
            "wer_caption_vs_asr": w_cap_asr,
            "wer_timed_vs_asr": w_tim_asr,
            "wer_caption_vs_timed": w_cap_tim,
            "char_ratio_caption_vs_timed": cr_cap_tim,
        })

        print(f"\n[{rec['_group']}/{rec['image_id']}#{rec['annotator_id']}]")
        print(f"  caption       : {rec['caption']}")
        print(f"  timed-joined  : {join_timed(rec['timed_caption'])}")
        print(f"  parakeet ASR  : {asr}")
        print(f"  WER caption↔ASR={w_cap_asr:.3f}  timed↔ASR={w_tim_asr:.3f}  "
              f"caption↔timed={w_cap_tim:.3f}  (charsim cap↔timed={cr_cap_tim:.3f})")

    print("\n" + "=" * 80)
    print("Aggregate (mean over samples)")
    print("=" * 80)

    def avg(xs): return sum(xs) / len(xs) if xs else 0.0

    print(f"  WER caption ↔ parakeet : {avg(agg['wer_caption_vs_asr']):.3f}")
    print(f"  WER timed   ↔ parakeet : {avg(agg['wer_timed_vs_asr']):.3f}")
    print(f"  WER caption ↔ timed    : {avg(agg['wer_caption_vs_timed']):.3f}")
    print(f"  char-sim caption ↔ timed: {avg(agg['char_ratio_caption_vs_timed']):.3f}")

    print("\nInterpretation guide:")
    print("  caption↔timed near 0   → caption IS the timed-caption concatenation")
    print("  caption↔timed > 0.05   → caption was edited away from the spoken words")
    print("  caption↔ASR < ~0.20    → caption tracks the audio verbatim (ASR-level diff only)")
    print("  caption↔ASR  > ~0.30   → caption substantially rephrased / cleaned")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({
            "samples": results,
            "aggregate": {k: avg(v) for k, v in agg.items()},
        }, indent=2, ensure_ascii=False))
        print(f"\nSaved per-sample results to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
