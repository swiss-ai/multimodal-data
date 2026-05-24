#!/usr/bin/env python3
"""
Fallback index builder for collections with no metadata-image join.

aaa  — 8,504 images of archival audio/video tapes (cassettes, reels).
        Filenames encode collection ID, tape type and view angle.
        No online_media in metadata. Build records from filename parsing.

nmai — 255 images of Native American artifacts with sequential NMAI-NNN IDs
        that don't match metadata record_IDs. Build minimal records.

Both go into tier2 for VLM recaptioning. Appended to index_art.parquet.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

INDEX_DIR = Path("/tmp/toolbox/smithsonian/data/indices")
MEDIA_ROOT = Path("/path/to/data/vision-datasets/raw/cooldown/s3___smithsonian___OpenAccess/media")
META_ROOT = Path("/path/to/data/vision-datasets/raw/cooldown/s3___smithsonian___OpenAccess/metadata/edan")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(INDEX_DIR / "fallback.log"),
    ],
)
log = logging.getLogger("fallback")

EMPTY_ROW = dict(
    date="",
    creator="",
    medium="",
    place="",
    object_type="",
    topics=[],
    credit_line="",
    data_source="",
    set_name="",
    notes_raw=[],
    scopecontent="",
    alt_text="",
    ext_descr="",
    taxonomic_name="",
    identifier="",
    guid="",
    license="CC0",
    all_idsIds=[],
)

# ── AAA ────────────────────────────────────────────────────────────────────────
# Filename pattern: AAA-{collection_id}_{recording_id}_{medium}_{seq_of_total}_{view}.jpg
# e.g. AAA-aaa_Abbens64_reel_1of1_front.jpg
#      AAA-aaa_abbot81_cassette_1of2_side.jpg

_AAA_RE = re.compile(
    r"AAA-(?P<coll>[^_]+)_(?P<rec>[^_]+)_(?P<medium>reel|cassette|cass|tape|film|disc|disk|card|box|photo|slide)s?"
    r"(?:_(?P<seq>\d+of\d+))?_(?P<view>front|back|side|top|bottom|label|cover|inner|outer|reel|a|b)",
    re.IGNORECASE,
)


def parse_aaa_filename(stem: str) -> dict:
    """Return a metadata dict derived purely from the AAA filename stem."""
    m = _AAA_RE.match(stem)
    if m:
        rec = m.group("rec")
        medium = m.group("medium").lower()
        seq = m.group("seq") or "1of1"
        view = m.group("view").lower()
        medium_full = {
            "reel": "reel-to-reel audio tape",
            "cass": "audio cassette tape",
            "cassette": "audio cassette tape",
            "tape": "audio/video tape",
            "film": "film reel",
            "disc": "disc recording",
            "disk": "disk recording",
            "card": "index card",
            "box": "storage box",
            "photo": "photograph",
            "slide": "slide",
        }.get(medium, medium)
        title = f"{medium_full.capitalize()} — {rec} ({seq}) — {view} view"
        obj_type = "archival audio/video media"
        description = (
            f"{view.capitalize()} view of a {medium_full} labeled '{rec}' "
            f"({seq.replace('of', ' of ')} reels/tapes). "
            f"From the Archives of American Art oral history collection."
        )
    else:
        # Fallback: use full stem
        title = stem.replace("AAA-", "").replace("_", " ")
        obj_type = "archival document"
        description = f"Archival item from the Archives of American Art: {title}."

    return {"title": title, "object_type": obj_type, "_description": description}


def build_aaa_records() -> list[dict]:
    media_dir = MEDIA_ROOT / "aaa"
    if not media_dir.exists():
        log.warning("No media/aaa directory")
        return []

    records = []
    for fname in sorted(os.listdir(media_dir)):
        if not fname.lower().endswith(".jpg"):
            continue
        stem = fname[:-4]
        fpath = str(media_dir / fname)
        meta = parse_aaa_filename(stem)

        records.append(
            {
                **EMPTY_ROW,
                "collection": "aaa",
                "media_dir": "aaa",
                "tier_path": "tier2/history/aaa",
                "record_id": stem,
                "type": "filename_only",
                "title": meta["title"],
                "object_type": meta["object_type"],
                "data_source": "Archives of American Art, Smithsonian Institution",
                "scopecontent": meta["_description"],  # used as best-effort caption
                "primary_idsId": stem,
                "all_idsIds": [stem],
                "image_path": fpath,
            }
        )

    log.info(f"AAA fallback: {len(records)} records")
    return records


# ── NMAI ───────────────────────────────────────────────────────────────────────
# Filenames: NMAI-001.jpg, NMAI-001-000001.jpg, NMAI-001-000002.jpg …
# Try to match via nmai metadata — some records DO have online_media with ARK idsIds,
# but those ARK ids don't map to filenames.
# Best effort: scan metadata for any descriptor and attach to the sequential slot,
# or create unmatched minimal records.


def build_nmai_records() -> list[dict]:
    media_dir = MEDIA_ROOT / "nmai"
    if not media_dir.exists():
        log.warning("No media/nmai directory")
        return []

    # Group files by base number (NMAI-001 is primary, NMAI-001-000001 is a view)
    groups: dict[str, list[str]] = {}
    for fname in sorted(os.listdir(media_dir)):
        if not fname.lower().endswith(".jpg"):
            continue
        stem = fname[:-4]
        m = re.match(r"(NMAI-\d+)(?:-\d+)?$", stem)
        if m:
            base = m.group(1)
            groups.setdefault(base, []).append(str(media_dir / fname))

    log.info(f"NMAI: {len(groups)} unique objects, {sum(len(v) for v in groups.values())} images")

    # Try to pull any metadata we can from nmai shards (some records have basic info)
    # Index by sequential number if possible
    meta_by_recid: dict[str, dict] = {}
    meta_dir = META_ROOT / "nmai"
    if meta_dir.exists():
        for fn in sorted(os.listdir(meta_dir)):
            fpath = meta_dir / fn
            if fn == "index.txt" or os.path.getsize(fpath) == 0:
                continue
            with open(fpath) as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        d = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    dnr = d.get("content", {}).get("descriptiveNonRepeating", {})
                    rec_id = dnr.get("record_ID", "")
                    ft = d.get("content", {}).get("freetext", {})
                    title = (d.get("title") or "").strip()
                    if rec_id and title:
                        meta_by_recid[rec_id] = {
                            "title": title,
                            "date": " | ".join(e["content"] for e in ft.get("date", []) if e.get("content")),
                            "place": " | ".join(e["content"] for e in ft.get("place", []) if e.get("content")),
                            "medium": " | ".join(
                                e["content"] for e in ft.get("physicalDescription", []) if e.get("content")
                            ),
                            "object_type": " | ".join(
                                e["content"] for e in ft.get("objectType", []) if e.get("content")
                            ),
                            "notes_raw": ft.get("notes", []),
                        }

    log.info(f"NMAI: loaded {len(meta_by_recid)} metadata records")

    records = []
    for base, paths in sorted(groups.items()):
        # Prefer primary (no suffix) as main image
        primary = next((p for p in paths if re.search(r"NMAI-\d+\.jpg$", p)), paths[0])
        stem = Path(primary).stem

        # Try to find matching metadata (usually won't work for this collection)
        matched_meta = None
        num = re.search(r"NMAI-(\d+)", base)
        if num:
            padded = f"NMAI_{int(num.group(1)):06d}"
            matched_meta = meta_by_recid.get(padded) or meta_by_recid.get(f"NMAI_{num.group(1)}")

        if matched_meta:
            title = matched_meta["title"]
            date = matched_meta["date"]
            place = matched_meta["place"]
            medium = matched_meta["medium"]
            object_type = matched_meta["object_type"] or "Native American artifact"
            notes_raw = matched_meta["notes_raw"]
        else:
            title = f"Native American artifact {base}"
            date = ""
            place = ""
            medium = ""
            object_type = "Native American artifact"
            notes_raw = []

        records.append(
            {
                **EMPTY_ROW,
                "collection": "nmai",
                "media_dir": "nmai",
                "tier_path": "tier2/other/nmai",
                "record_id": stem,
                "type": "filename_only",
                "title": title,
                "date": date,
                "place": place,
                "medium": medium,
                "object_type": object_type,
                "notes_raw": notes_raw,
                "data_source": "National Museum of the American Indian, Smithsonian Institution",
                "primary_idsId": stem,
                "all_idsIds": [Path(p).stem for p in paths],
                "image_path": primary,
            }
        )

    log.info(f"NMAI fallback: {len(records)} records")
    return records


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    art_path = INDEX_DIR / "index_art.parquet"
    if not art_path.exists():
        log.error("index_art.parquet not found — run 01_build_index.py first")
        sys.exit(1)

    existing = pd.read_parquet(art_path)
    log.info(f"Existing art index: {len(existing):,} records")

    # Idempotency: skip collections already present
    existing_colls = set(existing["collection"].unique()) if "collection" in existing.columns else set()
    aaa_records = build_aaa_records() if "aaa" not in existing_colls else []
    nmai_records = build_nmai_records() if "nmai" not in existing_colls else []
    if "aaa" in existing_colls:
        log.info("AAA already in index — skipping")
    if "nmai" in existing_colls:
        log.info("NMAI already in index — skipping")
    all_new = aaa_records + nmai_records

    if not all_new:
        log.info("No fallback records to add.")
        return

    new_df = pd.DataFrame(all_new)
    combined = pd.concat([existing, new_df], ignore_index=True)
    # Drop any struct columns that pyarrow can't handle (empty dicts / no child fields)
    for col in ["renders"]:
        if col in combined.columns:
            combined = combined.drop(columns=[col])
    combined.to_parquet(art_path, index=False)
    log.info(f"Appended {len(all_new):,} fallback records → {art_path}")
    log.info(f"New art index total: {len(combined):,} records")

    tier_counts = combined.groupby("tier_path").size()
    log.info(f"\nFull art index breakdown:\n{tier_counts.to_string()}")


if __name__ == "__main__":
    main()
