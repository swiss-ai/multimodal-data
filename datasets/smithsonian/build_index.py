#!/usr/bin/env python3
"""
Phase 0 — Build join indices for the Smithsonian WebDataset.

Scans all metadata/edan shards and resolves each record to a local image file.
Outputs one Parquet file per collection group:
  - index_art.parquet   (all art/history/culture/design collections)
  - index_nmnh.parquet  (all NMNH natural-history divisions)
  - index_3d.parquet    (3D object renders)
"""

import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW = Path("/path/to/data/vision-datasets/raw/cooldown/s3___smithsonian___OpenAccess")
MEDIA_ROOT = RAW / "media"
META_ROOT = RAW / "metadata/edan"
INDEX_DIR = Path("/tmp/toolbox/smithsonian/data/indices")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(INDEX_DIR / "build_index.log"),
    ],
)
log = logging.getLogger("build_index")

# ── Collection config ──────────────────────────────────────────────────────────
# meta_coll → (media_subdir, tier_path)
ART_COLLS = {
    "saam": ("saam", "tier1/saam"),
    "nmaahc": ("nmaahc", "tier1/nmaahc"),
    "nmafa": ("nmafa", "tier1/nmafa"),
    "npg": ("npg", "tier1/npg"),
    "nmah": ("nmah", "tier2/history/nmah"),
    "cfch": ("cfch", "tier2/history/cfch"),
    "nasm": ("nasm", "tier2/history/nasm"),
    "aaa": ("aaa", "tier2/history/aaa"),
    "fs": ("fs", "tier2/art/fs"),
    "hmsg": ("hmsg", "tier2/art/hmsg"),
    "chsdm": ("chsdm", "tier2/design/chsdm"),
    "chndm": (
        "chndm",
        "tier2/design/chndm",
    ),  # images downloaded by 04_download_chndm.py
    "npm": ("npm", "tier2/other/npm"),
    "sg": ("sg", "tier2/other/sg"),
    "nzp": ("nzp", "tier2/other/nzp"),
    "sia": ("sia", "tier2/other/sia"),
    "acm": ("acm", "tier2/other/acm"),
    "nmai": ("nmai", "tier2/other/nmai"),
    "fm": ("fm", "tier2/other/fm"),
    "sf": ("sf", "tier2/other/sf"),
}

NMNH_DIVS = [
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
]

# ── Helpers ────────────────────────────────────────────────────────────────────


def ft_get(ft, key):
    """Return '|'-joined content values for a freetext key."""
    return " | ".join(e["content"] for e in ft.get(key, []) if e.get("content", "").strip())


def ft_list(ft, key):
    return [e["content"] for e in ft.get(key, []) if e.get("content", "").strip()]


def build_media_set(media_dir: Path):
    """Return set of filename stems (without .jpg) present in a media directory."""
    if not media_dir.exists():
        return set()
    return {f[:-4] for f in os.listdir(media_dir) if f.lower().endswith(".jpg")}


def shards_of(meta_dir: Path):
    return [meta_dir / fn for fn in os.listdir(meta_dir) if fn != "index.txt" and os.path.getsize(meta_dir / fn) > 0]


# ── Art/history shard worker ───────────────────────────────────────────────────


def _parse_art_shard(args):
    coll, media_dir_name, tier_path, shard_path, media_stems = args
    records = []
    try:
        with open(shard_path) as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                rtype = d.get("type", "")
                content = d.get("content", {})
                ft = content.get("freetext", {})
                dnr = content.get("descriptiveNonRepeating", {})

                # ── Extract media idsIds ──────────────────────────────────────
                media_obj = dnr.get("online_media", {})
                media_items = media_obj.get("media", []) if isinstance(media_obj, dict) else []

                idsIds, alt_text, ext_descr = [], "", ""
                for m in media_items:
                    ids_id = m.get("idsId", "")
                    if ids_id:
                        idsIds.append(ids_id)
                    if not alt_text:
                        alt_text = m.get("altTextAccessibility", "")
                    if not ext_descr:
                        ext_descr = m.get("extDescrAccessibility", "")

                if not idsIds:
                    continue

                # ── Filter to locally present images ─────────────────────────
                present = [i for i in idsIds if i in media_stems]
                if not present:
                    continue

                # Prefer _1 / _001 as primary view
                primary = present[0]
                for ids_id in present:
                    tail = ids_id.rsplit("_", 1)[-1] if "_" in ids_id else ""
                    if tail in ("1", "001"):
                        primary = ids_id
                        break

                # ── Text fields ───────────────────────────────────────────────
                notes = ft.get("notes", [])
                scopecontent = ""
                if rtype == "ead_component":
                    for fk in ("scopecontent", "abstract", "bioghist"):
                        v = ft_get(ft, fk)
                        if v:
                            scopecontent = v
                            break

                records.append(
                    {
                        "collection": coll,
                        "media_dir": media_dir_name,
                        "tier_path": tier_path,
                        "record_id": d.get("id", ""),
                        "type": rtype,
                        "title": (d.get("title") or "").strip(),
                        "date": ft_get(ft, "date"),
                        "creator": ft_get(ft, "name"),
                        "medium": ft_get(ft, "physicalDescription"),
                        "place": ft_get(ft, "place"),
                        "object_type": ft_get(ft, "objectType") or ft_get(ft, "type"),
                        "topics": ft_list(ft, "topic"),
                        "credit_line": ft_get(ft, "creditLine"),
                        "data_source": ft_get(ft, "dataSource"),
                        "set_name": ft_get(ft, "setName"),
                        "notes_raw": notes,
                        "scopecontent": scopecontent,
                        "alt_text": alt_text,
                        "ext_descr": ext_descr,
                        "guid": dnr.get("guid", ""),
                        "license": "CC0",
                        "primary_idsId": primary,
                        "all_idsIds": present,
                        "image_path": str(MEDIA_ROOT / media_dir_name / f"{primary}.jpg"),
                    }
                )
    except Exception as e:
        log.warning(f"Shard error {shard_path}: {e}")
    return records


def process_art_collection(coll):
    media_dir_name, tier_path = ART_COLLS[coll]
    meta_dir = META_ROOT / coll
    media_dir = MEDIA_ROOT / media_dir_name

    if not meta_dir.exists():
        log.warning(f"No metadata dir: {meta_dir}")
        return []

    media_stems = build_media_set(media_dir)
    shards = shards_of(meta_dir)
    log.info(f"{coll}: {len(media_stems):,} images, {len(shards)} shards")

    shard_args = [(coll, media_dir_name, tier_path, s, media_stems) for s in shards]

    all_records = []
    with ThreadPoolExecutor(max_workers=16) as exe:
        for batch in exe.map(_parse_art_shard, shard_args):
            all_records.extend(batch)

    log.info(f"{coll}: {len(all_records):,} matched records")
    return all_records


# ── NMNH shard worker ──────────────────────────────────────────────────────────


def _parse_nmnh_shard(args):
    div, shard_path, catalog_map = args
    records = []
    try:
        with open(shard_path) as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                content = d.get("content", {})
                ft = content.get("freetext", {})
                dnr = content.get("descriptiveNonRepeating", {})

                # Try identifier → catalog number
                ident_raw = ft_get(ft, "identifier")
                if not ident_raw:
                    continue

                # Extract all numeric substrings and try each
                candidates = re.findall(r"\d+", ident_raw)
                matched_paths = []
                matched_cat = None
                for num_str in candidates:
                    num = num_str.lstrip("0") or "0"
                    paths = catalog_map.get(num, [])
                    if paths:
                        matched_paths = paths
                        matched_cat = num_str
                        break

                if not matched_paths:
                    continue

                tax = ft_get(ft, "taxonomicName")
                notes = ft.get("notes", [])

                # Prefer primary view (_1 or no suffix)
                def sort_key(p):
                    name = Path(p).stem
                    return 1 if re.search(r"-\d+$", name) else 0

                matched_paths.sort(key=sort_key)
                primary = matched_paths[0]

                records.append(
                    {
                        "collection": div,
                        "media_dir": "nmnh",
                        "tier_path": f"tier3/nmnh/{div.replace('nmnh', '')}",
                        "record_id": d.get("id", ""),
                        "type": d.get("type", ""),
                        "title": (d.get("title") or "").strip(),
                        "date": ft_get(ft, "date"),
                        "creator": ft_get(ft, "name"),
                        "medium": ft_get(ft, "physicalDescription"),
                        "place": ft_get(ft, "place"),
                        "object_type": "natural history specimen",
                        "topics": [],
                        "credit_line": "",
                        "data_source": ft_get(ft, "dataSource"),
                        "set_name": ft_get(ft, "setName"),
                        "notes_raw": notes,
                        "scopecontent": "",
                        "alt_text": "",
                        "ext_descr": "",
                        "taxonomic_name": tax,
                        "identifier": matched_cat or ident_raw,
                        "guid": dnr.get("guid", ""),
                        "license": "CC0",
                        "primary_idsId": Path(primary).stem,
                        "all_idsIds": [Path(p).stem for p in matched_paths],
                        "image_path": primary,
                    }
                )
    except Exception as e:
        log.warning(f"NMNH shard error {shard_path}: {e}")
    return records


def process_nmnh():
    nmnh_media = MEDIA_ROOT / "nmnh"
    if not nmnh_media.exists():
        log.error("No media/nmnh directory")
        return []

    log.info("Building NMNH catalog → file map (1.2M files)…")
    catalog_map: dict[str, list[str]] = {}
    for fname in tqdm(os.listdir(nmnh_media), desc="NMNH media scan"):
        if not fname.lower().endswith(".jpg"):
            continue
        m = re.match(r"NMNH-0*(\d+)", fname)
        if m:
            num = m.group(1)
            fpath = str(nmnh_media / fname)
            catalog_map.setdefault(num, []).append(fpath)
    log.info(f"NMNH catalog map: {len(catalog_map):,} unique catalog numbers")

    all_records = []
    for div in NMNH_DIVS:
        meta_dir = META_ROOT / div
        if not meta_dir.exists():
            log.warning(f"No metadata dir for {div}")
            continue
        shards = shards_of(meta_dir)
        shard_args = [(div, s, catalog_map) for s in shards]
        log.info(f"{div}: {len(shards)} shards")

        div_records = []
        with ThreadPoolExecutor(max_workers=16) as exe:
            for batch in exe.map(_parse_nmnh_shard, shard_args):
                div_records.extend(batch)

        log.info(f"{div}: {len(div_records):,} matched records")
        all_records.extend(div_records)

    return all_records


# ── 3D worker ──────────────────────────────────────────────────────────────────


def process_3d():
    media_3d = MEDIA_ROOT / "3d"
    if not media_3d.exists():
        log.warning("No media/3d directory")
        return []

    # Build ocio_dpo3d metadata lookup by identifier
    ocio_meta = {}
    ocio_dir = META_ROOT / "ocio_dpo3d"
    if ocio_dir.exists():
        for fn in os.listdir(ocio_dir):
            fpath = ocio_dir / fn
            if fn == "index.txt" or os.path.getsize(fpath) == 0:
                continue
            with open(fpath) as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        d = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    ft = d.get("content", {}).get("freetext", {})
                    dnr = d.get("content", {}).get("descriptiveNonRepeating", {})
                    rec_id = dnr.get("record_ID", "")
                    ident = ft_get(ft, "identifier")
                    notes = ft.get("notes", [])
                    entry = {
                        "title": (d.get("title") or "").strip(),
                        "notes_raw": notes,
                        "taxonomic_name": ft_get(ft, "taxonomicName"),
                        "date": ft_get(ft, "date"),
                        "place": ft_get(ft, "place"),
                        "data_source": ft_get(ft, "dataSource"),
                        "identifier": ident,
                        "record_id": d.get("id", ""),
                    }
                    ocio_meta[rec_id] = entry
                    # Also index by catalog number for fuzzy matching
                    for num in re.findall(r"\d+", ident):
                        ocio_meta[f"num:{num}"] = entry

    log.info(f"ocio_dpo3d: {len(ocio_meta)} metadata entries")

    records = []
    uuids = [u for u in os.listdir(media_3d) if (media_3d / u).is_dir()]
    log.info(f"3D: {len(uuids)} UUID folders")

    for uuid in uuids:
        uuid_dir = media_3d / uuid
        # Check for scene.svx.json
        svx_path = uuid_dir / "scene.svx.json"
        meta_entry = {}
        if svx_path.exists():
            try:
                svx = json.loads(svx_path.read_text())
                # Try to extract catalog reference from svx
                svx_str = json.dumps(svx)
                # Look for dpo_3d_XXXXXX pattern
                m = re.search(r"dpo_3d_\d+", svx_str)
                if m and m.group(0) in ocio_meta:
                    meta_entry = ocio_meta[m.group(0)]
                else:
                    # Try numeric catalog matches
                    for num in re.findall(r"\b\d{4,8}\b", svx_str):
                        key = f"num:{num.lstrip('0') or '0'}"
                        if key in ocio_meta:
                            meta_entry = ocio_meta[key]
                            break
            except Exception:
                pass

        # Collect available renders (prefer high > medium > low > thumb)
        renders = {}
        for res in ("high", "medium", "low", "thumb"):
            p = uuid_dir / f"scene-image-{res}.jpg"
            if p.exists():
                renders[res] = str(p)

        if not renders:
            continue

        primary = renders.get("high") or next(iter(renders.values()))

        records.append(
            {
                "collection": "3d",
                "media_dir": "3d",
                "tier_path": "tier4/3d",
                "record_id": meta_entry.get("record_id", uuid),
                "type": "3d_render",
                "title": meta_entry.get("title", ""),
                "date": meta_entry.get("date", ""),
                "creator": "",
                "medium": "3D scan",
                "place": meta_entry.get("place", ""),
                "object_type": "3D museum specimen",
                "topics": [],
                "credit_line": "",
                "data_source": meta_entry.get("data_source", "Smithsonian"),
                "set_name": "",
                "notes_raw": meta_entry.get("notes_raw", []),
                "scopecontent": "",
                "alt_text": "",
                "ext_descr": "",
                "taxonomic_name": meta_entry.get("taxonomic_name", ""),
                "identifier": meta_entry.get("identifier", uuid),
                "guid": uuid,
                "license": "CC0",
                "primary_idsId": f"3d_{uuid}",
                "all_idsIds": [f"3d_{uuid}_{r}" for r in renders],
                "image_path": primary,
                "renders": renders,
            }
        )

    log.info(f"3D: {len(records)} records with images")
    return records


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    # ── Art/history/culture/design ───────────────────────────────────────────
    art_out = INDEX_DIR / "index_art.parquet"
    if art_out.exists():
        log.info(f"Skipping art index (already exists): {art_out}")
    else:
        log.info(
            f"Processing {len(ART_COLLS)} art/history collections with {min(len(ART_COLLS), cpu_count())} workers…"
        )
        workers = min(len(ART_COLLS), 20)
        with Pool(workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(process_art_collection, list(ART_COLLS.keys())),
                    total=len(ART_COLLS),
                    desc="Art collections",
                )
            )
        art_records = [r for batch in results for r in batch]
        log.info(f"Art total: {len(art_records):,} records")
        df = pd.DataFrame(art_records)
        df.to_parquet(art_out, index=False)
        log.info(f"Saved → {art_out}")

    # ── NMNH natural history ─────────────────────────────────────────────────
    nmnh_out = INDEX_DIR / "index_nmnh.parquet"
    if nmnh_out.exists():
        log.info(f"Skipping NMNH index (already exists): {nmnh_out}")
    else:
        log.info("Processing NMNH divisions…")
        nmnh_records = process_nmnh()
        log.info(f"NMNH total: {len(nmnh_records):,} records")
        df = pd.DataFrame(nmnh_records)
        df.to_parquet(nmnh_out, index=False)
        log.info(f"Saved → {nmnh_out}")

    # ── 3D renders ───────────────────────────────────────────────────────────
    d3_out = INDEX_DIR / "index_3d.parquet"
    if d3_out.exists():
        log.info(f"Skipping 3D index (already exists): {d3_out}")
    else:
        log.info("Processing 3D renders…")
        d3_records = process_3d()
        log.info(f"3D total: {len(d3_records):,} records")
        df = pd.DataFrame(d3_records)
        df.to_parquet(d3_out, index=False)
        log.info(f"Saved → {d3_out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    for label, path in [("Art/history", art_out), ("NMNH", nmnh_out), ("3D", d3_out)]:
        if path.exists():
            df = pd.read_parquet(path, columns=["collection", "tier_path"])
            log.info(f"\n{label} breakdown:\n{df.groupby('tier_path').size().to_string()}")


if __name__ == "__main__":
    main()
