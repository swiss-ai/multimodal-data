"""
Build a single narrations.parquet from two ego4d annotation files.

Sources:
  - all_narrations_redacted.json  (8922 videos, PII redacted, more narrations)
  - narration.json                (9611 videos; only 689 exclusive videos used)

Output columns:
  key        – "{video_uid}_{i:06d}", unique, used as image/caption filename stem
  video_uid  – ego4d video identifier
  timestamps – list[float], seconds into the full-scale video:
                 narrations → 1 timestamp (midpoint of annotation window)
                 summaries  → 5 timestamps at 1/8–5/8 of interval
                              (only midpoint if interval < SUMMARY_MIN_SEC)
  caption    – cleaned text, ready for training
  type       – "narration" or "summary"

Text cleaning (minimal):
  Any #tag (incl. #C, #O, #unsure, #summary) → removed
  Standalone "C" (whole word) → randomly replaced with one of C_ALTERNATIVES,
    consistently within each text (all occurrences get the same substitute)

Summaries: non-overlapping chunk summaries kept per video.
  Overlapping ones (same chunk, multiple annotators) → keep longest text.

Deduplication: (video_uid, caption, round(timestamp_0)) per-second granularity.
"""

import os
import random
import re

import orjson
import polars as pl

NARRATION_JSON = "/path/to/data/vision-datasets/ego4d/v2/annotations/narration.json"
ALL_NARRATIONS_JSON = "/path/to/data/vision-datasets/ego4d/v2/annotations/all_narrations_redacted.json"
OUT_PARQUET = "/tmp/ego4d/narrations.parquet"

SUMMARY_MIN_SEC = 20.0  # intervals shorter than this → single midpoint timestamp

C_ALTERNATIVES = (
    100 * ["the actor"]
    + 50 * ["the user"]
    + 5 * ["the subject"]
    + 5 * ["the participant"]
    + 5 * ["the individual"]
    + 1 * ["the ego"]
)

_rng = random.Random()  # unseeded — variety across runs is fine for training data


# ── Text cleaning ─────────────────────────────────────────────────────────────

_RE_ANY_TAG = re.compile(r"#\w+\s*")
_RE_C = re.compile(r"(?<![A-Za-z])C(?![A-Za-z])")
_RE_SPACES = re.compile(r"\s{2,}")


def clean_text(raw: str) -> str:
    """Minimal cleaning: strip #tags, replace C consistently with a random alt."""
    text = raw.strip()
    text = _RE_ANY_TAG.sub("", text)  # remove all #tags
    text = _RE_C.sub(_rng.choice(C_ALTERNATIVES), text)  # consistent sub per call
    text = _RE_SPACES.sub(" ", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


# ── Timestamp helpers ─────────────────────────────────────────────────────────


def narration_timestamp(clip_start: float, clip_end: float) -> list:
    return [(clip_start + clip_end) * 0.5]


def summary_timestamps(start: float, end: float) -> list:
    """5 timestamps at 1/8…5/8, or just the midpoint for very short intervals."""
    duration = end - start
    if duration < SUMMARY_MIN_SEC:
        return [(start + end) * 0.5]
    return [start + duration * k / 8 for k in range(1, 6)]


# ── Summary helpers ───────────────────────────────────────────────────────────


def select_nonoverlapping(summaries: list, start_key: str, end_key: str, text_key: str) -> list:
    """Keep one summary per non-overlapping time chunk (longest text wins)."""
    if not summaries:
        return []
    by_start = sorted(summaries, key=lambda s: s[start_key])
    selected, cluster = [], [by_start[0]]
    for s in by_start[1:]:
        if s[start_key] < max(x[end_key] for x in cluster):
            cluster.append(s)
        else:
            selected.append(max(cluster, key=lambda x: len(x[text_key])))
            cluster = [s]
    selected.append(max(cluster, key=lambda x: len(x[text_key])))
    return selected


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("Loading all_narrations_redacted.json …", flush=True)
    with open(ALL_NARRATIONS_JSON, "rb") as f:
        all_data = orjson.loads(f.read())

    print("Loading narration.json …", flush=True)
    with open(NARRATION_JSON, "rb") as f:
        narr_data = orjson.loads(f.read())

    all_videos = all_data["videos"]
    redacted_vids = set(all_videos.keys())
    exclusive = {uid: v for uid, v in narr_data.items() if uid not in redacted_vids}

    # ── Phase 1: pre-collect and clean summaries per video ────────────────────
    # Summaries are cleaned once here; the same cleaned text is reused both as
    # summary rows in the parquet and as context for short narrations.
    print("Pre-collecting summaries …", flush=True)
    video_summaries: dict[str, list] = {}

    for video_uid, v in all_videos.items():
        sums = select_nonoverlapping(v.get("summaries", []), "start_time", "end_time", "text")
        video_summaries[video_uid] = [
            {
                "caption": clean_text(s["text"]),
                "start": float(s["start_time"]),
                "end": float(s["end_time"]),
            }
            for s in sums
        ]

    for video_uid, v in exclusive.items():
        all_sums = []
        for pass_key in ("narration_pass_1", "narration_pass_2"):
            all_sums.extend(v.get(pass_key, {}).get("summaries", []))
        sums = select_nonoverlapping(all_sums, "start_sec", "end_sec", "summary_text")
        video_summaries[video_uid] = [
            {
                "caption": clean_text(s["summary_text"]),
                "start": float(s["start_sec"]),
                "end": float(s["end_sec"]),
            }
            for s in sums
        ]

    # ── Phase 2 & 3: narrations + summaries → buckets ────────────────────────
    buckets: dict[str, list] = {}
    seen: set[tuple] = set()

    def add(video_uid: str, timestamps: list, caption: str, kind: str):
        buckets.setdefault(video_uid, []).append((timestamps, caption, kind))

    # all_narrations_redacted — narrations
    print(
        f"Processing narrations: all_narrations_redacted ({len(all_videos)} videos) …",
        flush=True,
    )
    for video_uid, v in all_videos.items():
        for narr in v.get("narrations", []):
            ts = narration_timestamp(
                float(narr.get("_clip_time_start", narr["time"])),
                float(narr["time"]),
            )
            caption = clean_text(narr["text"])
            dk = (video_uid, caption, round(ts[0]))
            if dk in seen:
                continue
            seen.add(dk)
            add(video_uid, ts, caption, "narration")

    # narration.json exclusive — narrations
    print(f"Processing narrations: exclusive videos ({len(exclusive)}) …", flush=True)
    for video_uid, v in exclusive.items():
        for pass_key in ("narration_pass_1", "narration_pass_2"):
            for narr in v.get(pass_key, {}).get("narrations", []):
                ts = [float(narr["timestamp_sec"])]
                caption = clean_text(narr["narration_text"])
                dk = (video_uid, caption, round(ts[0]))
                if dk in seen:
                    continue
                seen.add(dk)
                add(video_uid, ts, caption, "narration")

    # summaries (both sources)
    print("Processing summaries …", flush=True)
    for video_uid, sums in video_summaries.items():
        for s in sums:
            ts = summary_timestamps(s["start"], s["end"])
            dk = (video_uid, s["caption"], round(ts[0]))
            if dk in seen:
                continue
            seen.add(dk)
            add(video_uid, ts, s["caption"], "summary")

    # ── Flatten + write ───────────────────────────────────────────────────────
    keys, video_uids, timestamps_col, captions, types = [], [], [], [], []
    for video_uid, entries in buckets.items():
        for i, (ts, caption, kind) in enumerate(entries):
            keys.append(f"{video_uid}_{i:06d}")
            video_uids.append(video_uid)
            timestamps_col.append(ts)
            captions.append(caption)
            types.append(kind)

    print(f"Total rows: {len(keys):,}", flush=True)

    df = pl.DataFrame(
        {
            "key": keys,
            "video_uid": video_uids,
            "timestamps": timestamps_col,
            "caption": captions,
            "type": types,
        },
        schema={
            "key": pl.String,
            "video_uid": pl.String,
            "timestamps": pl.List(pl.Float64),
            "caption": pl.String,
            "type": pl.String,
        },
    )

    print(f"Writing {OUT_PARQUET} …", flush=True)
    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    df.write_parquet(OUT_PARQUET, compression="zstd")
    print("Done.", flush=True)
    print(df.head(10))
    print(df.group_by("type").len().sort("type"))


if __name__ == "__main__":
    main()
