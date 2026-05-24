#!/usr/bin/env python3
"""
Phase 1 — Tier scoring and caption construction (CPU only).

Reads the index Parquet files produced by 01_build_index.py and:
  - Classifies each record into tier1/tier2/tier3/tier4
  - Assembles a training-ready caption (key.txt content)
  - Builds a VLM grounding prompt (stored in JSON, used in Phase 3)
  - Outputs augmented Parquet files ready for packing.
"""

import html
import json
import logging
import os
import re
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from tqdm import tqdm

MAX_WORKERS = min(256, cpu_count())

# ── Paths ──────────────────────────────────────────────────────────────────────
INDEX_DIR = Path(os.path.join(os.environ.get("SCRATCH_DIR", "/tmp"), "toolbox/smithsonian/data/indices"))
CAPTION_DIR = Path(os.path.join(os.environ.get("SCRATCH_DIR", "/tmp"), "toolbox/smithsonian/data/captions"))
CAPTION_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(CAPTION_DIR / "build_captions.log"),
    ],
)
log = logging.getLogger("build_captions")

# ── Tier-1 note label classification ──────────────────────────────────────────
# Labels whose content describes visual appearance → keep for tier-1 caption
VISUAL_LABELS = {
    "gallery label",
    "luce center label",
    "object description",
    "description",
    "label",
    "summary",
    "physical description",
    "curatorial remarks",
    "catalog note",
    "overview",
    "about this object",
    "extended description",
    "content",
    "notes",
    "caption",
    "image description",
    "visual description",
    "age",
    "material description",
}

# Labels that are NOT visual → always exclude from tier-1 caption body
NON_VISUAL_LABELS = {
    "exhibition history",
    "bibliography",
    "references",
    "provenance",
    "catalogue status",
    "research note",
    "condition report",
    "inscription",
    "marks",
    "funding",
    "rights",
    "label text",
    "publication history",
    "loan history",
    "appraisal",
    "publication",
    "catalogue number",
}


def classify_note_label(label: str) -> str:
    """Return 'visual', 'non_visual', or 'unknown'."""
    lw = label.lower().strip()
    if any(kw in lw for kw in VISUAL_LABELS):
        return "visual"
    if any(kw in lw for kw in NON_VISUAL_LABELS):
        return "non_visual"
    return "unknown"


# ── Text cleaning ──────────────────────────────────────────────────────────────
# Patterns to strip from notes
_ACCESSION_RE = re.compile(
    r"\b[A-Z]{2,10}-[\d.]+[\d](_\d+)?\b"  # SAAM-1950.2.14_3
    r"|\b\d{4}\.\d+\.\d+\b"  # 1950.2.14
    r"|\bcat(?:alogue|alog)?\.?\s*no\.?\s*[\d.]+",  # cat. no. 123
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_WS_RE = re.compile(r"\s+")
_TRAILING_CITE = re.compile(  # strip trailing exhibition / publication citations
    r"\n+(?:exhibition\s+label|exhibition\s+catalogue|from\s+the\s+collection"
    r"|published\s+in|cited\s+in|see\s+also)[^\n]*",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _TRAILING_CITE.sub("", text)
    text = _ACCESSION_RE.sub("", text)
    text = _MULTI_WS_RE.sub(" ", text)
    return text.strip()


def notes_to_text(notes_raw, mode="visual_only") -> str:
    """
    Convert raw notes list to cleaned text.
    mode='visual_only'  → only include entries whose label is visual or unknown
    mode='all'          → include everything
    """
    # Handle None, empty list, numpy arrays (from parquet), etc.
    if notes_raw is None:
        return ""
    try:
        if len(notes_raw) == 0:
            return ""
    except TypeError:
        return ""
    if isinstance(notes_raw, str):
        try:
            notes_raw = json.loads(notes_raw)
        except Exception:
            return clean_text(notes_raw)

    parts = []
    for entry in notes_raw:
        if not isinstance(entry, (dict,)):
            # numpy void or other mapping-like from parquet
            try:
                entry = dict(entry)
            except Exception:
                continue
        label = entry.get("label", "")
        content = entry.get("content", "").strip()
        if not content:
            continue
        if mode == "visual_only" and classify_note_label(label) == "non_visual":
            continue
        parts.append(content)
    return clean_text("\n\n".join(parts))


# ── Tier-1 caption assembly ────────────────────────────────────────────────────


def build_tier1_caption(row) -> str:
    """Return clean curator prose for tier-1 records."""
    # Priority: ext_descr (often a full machine-readable description) > notes > alt_text
    text = ""
    if row.get("ext_descr", ""):
        text = clean_text(row["ext_descr"])
    if len(text) < 80 and row.get("notes_raw"):
        text = notes_to_text(row["notes_raw"], mode="visual_only")
    if len(text) < 80 and row.get("alt_text", ""):
        text = clean_text(row["alt_text"])
    return text


# ── Tier-2 caption (best-effort metadata prose + grounding prompt) ─────────────


def build_tier2_caption(row) -> tuple[str, str]:
    """
    Returns (best_effort_caption, grounding_prompt).
    best_effort_caption: used as key.txt until VLM overwrites it.
    grounding_prompt: stored in key.json for the VLM recaptioning phase.
    """
    # Build grounding prompt for VLM
    lines = [f"Object: {row.get('title', '')}"]
    if row.get("object_type"):
        lines.append(f"Type: {row['object_type']}")
    if row.get("creator"):
        lines.append(f"Creator: {row['creator']}")
    if row.get("date"):
        lines.append(f"Date: {row['date']}")
    if row.get("medium"):
        lines.append(f"Medium / physical description: {row['medium']}")
    if row.get("place"):
        lines.append(f"Place of origin: {row['place']}")
    if row.get("data_source"):
        lines.append(f"Collection: {row['data_source']}")
    if row.get("topics"):
        topics = row["topics"]
        if isinstance(topics, str):
            try:
                topics = json.loads(topics)
            except Exception:
                topics = [topics]
        if topics:
            lines.append(f"Topics: {', '.join(topics)}")

    # Include any notes as additional context (capped at 600 chars)
    notes_ctx = notes_to_text(row.get("notes_raw"), mode="all")[:600]
    if notes_ctx:
        lines.append(f"\nAdditional context:\n{notes_ctx}")

    grounding = "\n".join(lines)

    # Best-effort caption: structured prose from available fields
    parts = []
    title = (row.get("title") or "").strip().strip(".")
    if title:
        parts.append(title + ".")
    if row.get("object_type"):
        parts.append(f"{row['object_type'].capitalize()}.")
    if row.get("creator") and row.get("date"):
        parts.append(f"Created by {row['creator']}, {row['date']}.")
    elif row.get("creator"):
        parts.append(f"Created by {row['creator']}.")
    elif row.get("date"):
        parts.append(f"Dated {row['date']}.")
    if row.get("medium"):
        parts.append(f"Medium: {row['medium']}.")
    if row.get("place"):
        parts.append(f"Place of origin: {row['place']}.")
    if row.get("data_source"):
        parts.append(f"From {row['data_source']}.")

    # Supplement with any short notes or scopecontent
    supp = (row.get("scopecontent") or "").strip()
    if not supp:
        supp = notes_to_text(row.get("notes_raw"), mode="visual_only")
    if supp and len(supp) > 40:
        parts.append(supp[:400])

    caption = " ".join(parts) if parts else title or "Museum object."
    return clean_text(caption), grounding


# ── Tier-3 NMNH template caption ──────────────────────────────────────────────


def build_tier3_caption(row) -> tuple[str, str]:
    """Return (template_caption, grounding_prompt) for NMNH specimens."""
    title = (row.get("title") or "").strip().strip(".")
    tax = (row.get("taxonomic_name") or "").strip()
    date = (row.get("date") or "").strip()
    place = (row.get("place") or "").strip()
    medium = (row.get("medium") or "").strip()
    div = (row.get("collection") or "").replace("nmnh", "").strip()

    parts = []
    if title and title.lower() not in ("indet. sp.", "indet", "unknown"):
        parts.append(title + ".")
    if tax:
        parts.append(f"Taxonomy: {tax}.")
    if date:
        parts.append(f"Collected {date}.")
    if place:
        parts.append(f"Location: {place}.")
    if medium:
        parts.append(f"{medium}.")

    # Add any useful notes (e.g. 'Age', 'Summary' from ocio_dpo3d / nmnh paleo)
    notes_text = notes_to_text(row.get("notes_raw"), mode="visual_only")
    if notes_text and len(notes_text) > 20:
        parts.append(notes_text[:300])

    caption = " ".join(parts) if parts else f"Natural history specimen from the {div} collection."

    grounding = (
        f"Object: {title or tax or 'Natural history specimen'}\n"
        f"Taxonomic classification: {tax}\n"
        f"Collection date: {date}\n"
        f"Collection location: {place}\n"
        f"Physical description: {medium}\n"
        f"Division: NMNH {div}"
    )

    return clean_text(caption), grounding


# ── Tier-4 3D renders ──────────────────────────────────────────────────────────


def build_tier4_caption(row) -> tuple[str, str]:
    title = (row.get("title") or "").strip()
    notes_text = notes_to_text(row.get("notes_raw"), mode="all")

    parts = []
    if title:
        parts.append(title + ".")
    if row.get("taxonomic_name"):
        parts.append(f"Taxonomy: {row['taxonomic_name']}.")
    if row.get("date"):
        parts.append(f"Age/date: {row['date']}.")
    if row.get("place"):
        parts.append(f"Origin: {row['place']}.")
    if notes_text:
        parts.append(notes_text[:400])
    parts.append("Rendered from a 3D scan of the museum specimen.")

    caption = " ".join(parts)
    grounding = (
        f"Object: {title}\n"
        f"Taxonomic name: {row.get('taxonomic_name', '')}\n"
        f"Age/date: {row.get('date', '')}\n"
        f"Origin: {row.get('place', '')}\n"
        f"Additional notes: {notes_text[:400]}\n"
        f"This is a rendered image from a 3D museum scan."
    )
    return clean_text(caption), grounding


# ── Tier classification ────────────────────────────────────────────────────────

TIER1_COLLS = {"saam", "nmaahc", "nmafa", "npg"}


def _to_list(val):
    """Convert numpy arrays / None to plain Python list for safe iteration."""
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return [val] if val else []
    try:
        return list(val)  # numpy ndarray, etc.
    except Exception:
        return []


def classify_and_caption(row) -> dict:
    """Add tier, caption_text, caption_source, grounding_prompt to row dict."""
    # Normalise list-valued fields so downstream code gets plain Python objects
    row = dict(row)
    row["notes_raw"] = _to_list(row.get("notes_raw"))
    row["topics"] = _to_list(row.get("topics"))
    row["all_idsIds"] = _to_list(row.get("all_idsIds"))

    coll = row.get("collection", "")
    rtype = row.get("type", "")
    notes_raw = row.get("notes_raw")
    tier_path = row.get("tier_path", "")

    if tier_path == "tier4/3d":
        caption, grounding = build_tier4_caption(row)
        return {
            **row,
            "tier": "tier4",
            "caption_text": caption,
            "caption_source": "template",
            "grounding_prompt": grounding,
        }

    if coll in NMNH_DIVS or tier_path.startswith("tier3"):
        caption, grounding = build_tier3_caption(row)
        return {
            **row,
            "tier": "tier3",
            "caption_text": caption,
            "caption_source": "template",
            "grounding_prompt": grounding,
        }

    # Art/history/design — check for tier-1 eligibility
    if coll in TIER1_COLLS and rtype == "edanmdm":
        visual_text = notes_to_text(notes_raw, mode="visual_only")
        ext = (row.get("ext_descr") or "").strip()
        best_text = ext if len(ext) > len(visual_text) else visual_text

        if len(best_text) >= 120:
            caption = build_tier1_caption(row)
            return {
                **row,
                "tier": "tier1",
                "caption_text": caption,
                "caption_source": "notes",
                "grounding_prompt": "",
            }

    # Anything else → tier2
    caption, grounding = build_tier2_caption(row)
    return {
        **row,
        "tier": "tier2",
        "caption_text": caption,
        "caption_source": "template+metadata",
        "grounding_prompt": grounding,
    }


NMNH_DIVS = {
    "nmnhbirds",
    "nmnhfishes",
    "nmnhbotany",
    "nmnhmammals",
    "nmnhherps",
    "nmnhento",
    "nmnhinv",
    "nmnhminsci",
    "nmnhpaleo",
    "nmnhanthro",
}

# ── Main ───────────────────────────────────────────────────────────────────────


def process_index(name: str, in_path: Path, out_path: Path):
    if out_path.exists():
        log.info(f"Skipping {name} captions (already exists): {out_path}")
        return

    log.info(f"Loading {in_path}…")
    df = pd.read_parquet(in_path)
    n = len(df)
    log.info(f"{name}: {n:,} records")

    rows = df.to_dict("records")

    # Use multiprocessing for large datasets; single-process for small ones
    if n > 50_000 and MAX_WORKERS > 1:
        log.info(f"Using {MAX_WORKERS} workers for {name}…")
        chunksize = max(1, n // (MAX_WORKERS * 8))
        with Pool(MAX_WORKERS) as pool:
            result_rows = list(
                tqdm(
                    pool.imap(classify_and_caption, rows, chunksize=chunksize),
                    total=n,
                    desc=f"Captions {name}",
                    unit="rec",
                )
            )
    else:
        result_rows = [classify_and_caption(r) for r in tqdm(rows, desc=f"Captions {name}", unit="rec")]

    out_df = pd.DataFrame(result_rows)
    out_df.to_parquet(out_path, index=False)
    log.info(f"Saved {len(out_df):,} rows → {out_path}")

    # Stats
    tier_counts = out_df.groupby(["tier", "caption_source"]).size()
    log.info(f"\n{name} tier breakdown:\n{tier_counts.to_string()}")

    # Caption length stats
    out_df["cap_len"] = out_df["caption_text"].str.len()
    log.info(f"\nCaption length stats:\n{out_df.groupby('tier')['cap_len'].describe().to_string()}")


def main():
    pairs = [
        (
            "Art/history",
            INDEX_DIR / "index_art.parquet",
            CAPTION_DIR / "captions_art.parquet",
        ),
        (
            "NMNH",
            INDEX_DIR / "index_nmnh.parquet",
            CAPTION_DIR / "captions_nmnh.parquet",
        ),
        ("3D", INDEX_DIR / "index_3d.parquet", CAPTION_DIR / "captions_3d.parquet"),
    ]
    for name, in_path, out_path in pairs:
        if not in_path.exists():
            log.warning(f"Index not found, skipping {name}: {in_path}")
            continue
        process_index(name, in_path, out_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
