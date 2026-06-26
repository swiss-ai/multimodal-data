"""Heuristic audit of V12 captions. Emits flagged rows + a random 200-sample for visual QA."""
import json
import random
import re
from pathlib import Path

import polars as pl

ROOT = Path("/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/owid")
OUT = ROOT / "qa_v12"
CAP = pl.read_parquet("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/owid___charts/grapher_charts_standalone_captioned.parquet")
SRC = pl.read_parquet("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/owid___charts/grapher_charts_standalone.parquet",
                     columns=["slug", "image_bytes", "title", "subtitle", "note"])

df = CAP.join(SRC, on="slug", how="inner")
print(f"joined rows: {df.height}")

# --- heuristics ---
BANNED = [
    r"\bsevere\b", r"\btragic\b", r"\bdevastat\w*", r"\btroubling\b", r"\balarming\b",
    r"\bdire\b", r"\bstark\b", r"\bdramatic\w*", r"\bstriking\b",
    r"\blikely\b", r"\bsuggests?\b", r"\bseems?\b", r"\bappears? to\b",
    r"\breflects?\b", r"\bindicates? that\b",
    r"\bwe (see|observe|notice|can see)\b", r"\byou can see\b",
    r"\bbecause\b", r"\bdue to\b", r"\bcaused by\b",
]
CHROME = [r"Data source:", r"CC BY", r"Our World in Data", r"ourworldindata\.org"]
MDLINK = r"\[[^\]]+\]\(#?[^\)]+\)"

def flags(row):
    c = row["caption"] or ""
    title = (row["title"] or "").strip()
    subtitle = (row["subtitle"] or "").strip()
    note = (row["note"] or "").strip()
    f = []
    for pat in BANNED:
        if re.search(pat, c, re.I):
            f.append(f"banned:{pat}")
    for pat in CHROME:
        if re.search(pat, c, re.I):
            f.append(f"chrome:{pat}")
    if re.search(MDLINK, c):
        f.append("mdlink")
    lines = [l for l in c.splitlines() if l.strip()]
    if not lines or not lines[0].strip():
        f.append("no_title_line")
    else:
        # title line should equal the metadata title (after md-strip)
        clean_t = re.sub(MDLINK, lambda m: m.group(0).split("]")[0][1:], title)
        if lines[0].strip().lower() != clean_t.lower():
            f.append("title_mismatch")
    wc = row["word_count"]
    if wc < 180:
        f.append(f"short:{wc}")
    if wc > 320:
        f.append(f"long:{wc}")
    # repeated phrases (4-gram cycle) = degeneration
    toks = c.lower().split()
    for i in range(len(toks) - 15):
        g = tuple(toks[i:i+4])
        if toks[i+4:i+8] == list(g) and toks[i+8:i+12] == list(g):
            f.append("repeat_4gram")
            break
    # Global-North binarization pattern: explicit claim like "Europe ... in the lightest/lowest"
    if re.search(r"(Europe|North America|Australia)[^.]{0,80}(lightest|lowest band|0[ -]?%?[ -]?10)", c, re.I):
        f.append("regional_binary")
    return f

records = []
for r in df.iter_rows(named=True):
    fl = flags(r)
    if fl:
        records.append({"slug": r["slug"], "word_count": r["word_count"], "flags": fl})

print(f"flagged: {len(records)} / {df.height} ({100*len(records)/df.height:.1f}%)")
# tally
from collections import Counter
tally = Counter()
for rec in records:
    for f in rec["flags"]:
        key = f.split(":")[0] if ":" in f and not f.startswith("banned") and not f.startswith("chrome") else f
        tally[key] += 1
print("flag tally:")
for k, v in tally.most_common(30):
    print(f"  {v:5d}  {k}")

(OUT / "flagged.json").write_text(json.dumps(records, indent=2))

# --- random sample 200 ---
random.seed(42)
all_slugs = df["slug"].to_list()
sample = random.sample(all_slugs, 200)
(OUT / "sample_slugs.json").write_text(json.dumps(sample, indent=2))
print(f"\nsampled 200 slugs -> {OUT/'sample_slugs.json'}")

# --- extract images + caption + metadata for sampled slugs ---
imgdir = OUT / "images"
imgdir.mkdir(exist_ok=True)
metadir = OUT / "meta"
metadir.mkdir(exist_ok=True)

sample_set = set(sample)
sub = df.filter(pl.col("slug").is_in(sample))
for r in sub.iter_rows(named=True):
    slug = r["slug"]
    (imgdir / f"{slug}.png").write_bytes(r["image_bytes"])
    (metadir / f"{slug}.json").write_text(json.dumps({
        "slug": slug,
        "title": r["title"],
        "subtitle": r["subtitle"],
        "note": r["note"],
        "caption": r["caption"],
        "word_count": r["word_count"],
    }, indent=2, ensure_ascii=False))

print(f"extracted {len(sample)} images + meta to {OUT}")
