"""Filter NASA shards down to knowledge-dense, PD-only cooldown training data.

Streaming design: reads one shard at a time, with ONLY the metadata columns
(no `image_bytes`), so total memory stays ~<2 GB regardless of corpus size.
Emits a slim `keep_ids.parquet` listing accepted nasa_ids; a separate export
step can join back to the shards to produce the final filtered dataset with
bytes.

Filters:
  1. Hard: `license == "PD"` (drops ESA-joint + UNCLEAR).
  2. Hard: len(description) >= 80.
  3. Hard: title != description (KSC/JSC event-photo duplication).
  4. Hard: description does NOT start with a PAO template
     (KENNEDY SPACE CENTER ..., ISS007-S-002, STS-114, etc.).
  5. Tiered decision:
       A (JPL/ARC/GRC/LRC/MSFC/AFRC/SSC/GSFC): keep if sci kw OR len>=200
       B (HQ):                                 keep if sci kw AND len>=300
       C (KSC/JSC):                            keep if sci kw AND len>=500
       X (other):                              keep if sci kw AND len>=200

Run:
    python3 filter_cooldown.py
"""
import json
from pathlib import Path

import polars as pl


SHARDS = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/web___nasa___images/shards")
OUT_DIR = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/nasa_cooldown")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns read from each shard — image_bytes is explicitly excluded.
COLS = ["nasa_id", "title", "description", "center", "license", "date_created"]

TIER_A = ["JPL", "ARC", "GRC", "LRC", "MSFC", "AFRC", "SSC", "GSFC"]
TIER_B = ["HQ"]
TIER_C = ["KSC", "JSC"]

# Polars regex (rust engine). Case-insensitive via (?i) inline flag.
SCIENCE_KW = (
    r"(?i)\b(wavelength|spectrum|spectra|instrument|resolution|acquired|"
    r"kilometers|miles across|light[- ]?year|galaxy|galaxies|nebula|supernova|"
    r"planet|planetary|surface|atmosphere|radiation|orbit|perspective view|"
    r"composition|formed|observed|latitude|longitude|hubble|jwst|spitzer|"
    r"chandra|webb|mars|jupiter|saturn|venus|mercury|pluto|asteroid|comet|"
    r"crater|volcano|volcanic|geology|geological|erosion|sediment|ocean|"
    r"cloud|storm|cyclone|hurricane|landsat|srtm|propulsion|aerodynamic|"
    r"wind tunnel|mach|supersonic|hypersonic|combustion|specimen|sample|"
    r"analysis|data|experiment|measurement|trajectory|solar|lunar|terrain)\b"
)

PAO_RE = (
    r"(?i)^([A-Z][A-Z ]+,\s*(FLA|FL|TEX|TX|VA|CA|MD|AL|CALIF|ALA|DC)\.?\s*[-—]|"
    r"iss\d{3}-[se]-\d{3}|sts-\d{2,3}|"
    r"kennedy space center|cape canaveral|johnson space center)"
)

# Biographical/career-essay patterns — caption describes a PERSON not an image.
# e.g. AFRC ECN-1203 Fred Haise: "was a research pilot... served as backup
# crewmember for Apollo 8... retired from NASA in 1979...". These teach the
# model to fabricate life histories for any portrait — pure noise for VLM.
BIO_RE = (
    r"(?i)\b(was born in|served as|flew on|retired (from|in)|"
    r"selected as an astronaut|backup crewmember|"
    r"received (his|her) (b\.?s\.?|m\.?s\.?|ph\.?d\.?|doctorate|bachelor|master|degree)|"
    r"earned (his|her) (degree|b\.?s\.?|m\.?s\.?|ph\.?d\.?|bachelor|master)|"
    r"from \d{4} to \d{4}|at the age of|"
    r"(is|was) a (research|former|retired|test|lead|chief|senior) (pilot|astronaut|engineer|scientist|director)|"
    r"joined (nasa|the agency)|"
    r"career (with|at) (nasa|the agency)|"
    r"spent \d+ (hours|days|years) in space)\b"
)

# Visual-grounding vocabulary: presence means the caption describes the image.
VISUAL_RE = (
    r"(?i)\b(shown (here|above|below|at)|visible (in|at)|pictured|appears "
    r"(in|at|on)|"
    r"in the (foreground|background|center|upper|lower|corner)|"
    r"this (image|view|photograph|photo|aerial|mosaic|scene|perspective)|"
    r"(wearing|holding|standing|seated|facing|posing)|"
    r"to the (left|right|north|south|east|west) of)\b"
)

# Person-name density: "Fred W. Haise Jr.", "John F. Kennedy"
NAME_RE = r"\b[A-Z][a-z]+\s+[A-Z]\.\s*([A-Z]\.\s*)?[A-Z][a-z]+\b"


def score_shard(path: Path) -> pl.DataFrame | None:
    """Read ONLY metadata columns and apply filters, return kept subset."""
    try:
        df = pl.read_parquet(path, columns=COLS)
    except Exception as e:
        print(f"  skip corrupt {path.name}: {str(e)[:80]}")
        return None

    raw_n = df.height
    df = df.filter(pl.col("license") == "PD")
    df = df.with_columns(pl.col("description").str.len_chars().alias("_dlen"))
    df = df.filter(pl.col("_dlen") >= 80)
    df = df.filter(
        pl.col("title").str.strip_chars() != pl.col("description").str.strip_chars()
    )
    df = df.with_columns(
        pl.col("description").str.contains(PAO_RE).alias("_pao"),
        pl.col("description").str.contains(SCIENCE_KW).alias("_sci"),
        # Biographical signal: >=2 career markers, low visual grounding, name-heavy.
        pl.col("description").str.count_matches(BIO_RE).alias("_bio_n"),
        pl.col("description").str.count_matches(VISUAL_RE).alias("_vis_n"),
        pl.col("description").str.count_matches(NAME_RE).alias("_name_n"),
        pl.when(pl.col("center").is_in(TIER_A)).then(pl.lit("A"))
          .when(pl.col("center").is_in(TIER_B)).then(pl.lit("B"))
          .when(pl.col("center").is_in(TIER_C)).then(pl.lit("C"))
          .otherwise(pl.lit("X")).alias("_tier"),
    ).with_columns(
        # Biographical caption: multiple career markers + zero visual grounding.
        # (Name count was unreliable — single-initial patterns like "Fred W.
        # Haise" confused the counter. Bio-markers + absent grounding is the
        # discriminative signal.)
        ((pl.col("_bio_n") >= 2) & (pl.col("_vis_n") == 0)).alias("_bio"),
    )
    keep = (
        (
            ((pl.col("_tier") == "A") & ~pl.col("_pao")
             & (pl.col("_sci") | (pl.col("_dlen") >= 200))) |
            ((pl.col("_tier") == "B") & pl.col("_sci") & (pl.col("_dlen") >= 300)) |
            ((pl.col("_tier") == "C") & pl.col("_sci") & (pl.col("_dlen") >= 500)
             & ~pl.col("_pao")) |
            ((pl.col("_tier") == "X") & pl.col("_sci") & (pl.col("_dlen") >= 200))
        )
        & ~pl.col("_bio")       # reject biographical-dump captions
    )
    kept = df.filter(keep)
    print(f"  {path.name}: raw={raw_n:>5}  pd_kept={kept.height:>5}  "
          f"({100*kept.height/max(raw_n,1):.0f}%)")
    return kept


def main() -> None:
    shards = sorted(SHARDS.glob("nasa_images_*.parquet"))
    print(f"=== scanning {len(shards)} shards (metadata-only; image_bytes excluded) ===")
    kept_parts = []
    for p in shards:
        sub = score_shard(p)
        if sub is not None:
            kept_parts.append(sub)

    if not kept_parts:
        raise SystemExit("no valid shards produced any keep rows")

    kept = pl.concat(kept_parts, how="vertical_relaxed")

    # Strip helper columns before persisting
    kept = kept.drop(["_dlen", "_pao", "_sci", "_tier",
                      "_bio", "_bio_n", "_vis_n", "_name_n"])

    out_ids = OUT_DIR / "keep_ids.parquet"
    kept.write_parquet(out_ids, compression="zstd")

    print(f"\n=== summary ===")
    print(f"  total kept ids: {kept.height:,}")
    print(f"  by center:")
    print(kept.group_by("center").agg([
        pl.len().alias("n"),
        pl.col("description").str.len_chars().mean().cast(pl.Int64).alias("mean_desc_len"),
    ]).sort("n", descending=True))

    summary = {
        "raw_shards_scanned": len(shards),
        "kept_rows": kept.height,
        "kept_by_center": kept.group_by("center").len().sort("len", descending=True).to_dicts(),
    }
    (OUT_DIR / "filter_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  -> wrote {out_ids}")
    print(f"  -> wrote {OUT_DIR/'filter_summary.json'}")
    print(f"\nNext step: an export script can stream each shard, inner-join with keep_ids,")
    print(f"and write final parquet with image_bytes — one shard at a time, bounded memory.")


if __name__ == "__main__":
    main()
