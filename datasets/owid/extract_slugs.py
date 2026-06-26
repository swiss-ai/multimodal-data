"""Parse OWID sitemap into a flat JSONL of things to download.

Emits three record types:
  - grapher:       one OWID-rendered chart -> fetch {slug}.png + {slug}.config.json
  - data_insight:  short article (title + paragraph + usually 1 embedded chart)
  - article:       longer narrative with embedded charts + surrounding prose

Usage:
    python extract_slugs.py --sitemap https://ourworldindata.org/sitemap.xml \
        --out /capstor/scratch/cscs/xyixuan/owid_data/slugs.jsonl
"""
import argparse
import json
import re
import urllib.request
from pathlib import Path


# Non-content pages we don't want to scrape as articles.
SKIP_TOPLEVEL = {
    "faqs", "about", "team", "jobs", "donate", "subscribe", "hiring-writer-2026",
    "privacy-policy", "cookie-notice", "organization", "funding", "teaching",
    "newsletters", "how-to-embed", "how-use-owid-for-high-school-teaching",
    "owid-for-teaching-feedback", "teaching-notes", "audience-survey-results",
    "instagram-in-spanish", "homepage-redesign", "introducing-new-search",
    "weve-improved-how-search-works-on-our-world-in-data",
    "redesigning-our-interactive-data-visualizations",
    "new-features-better-maps", "new-feature-embed-archived-charts",
    "owid-entry-redesign", "we-won-the-lovie-award", "owid-at-ycombinator",
    "introducing-new-search", "search",
}


def classify(url: str) -> tuple[str, str] | None:
    """Return (type, slug) or None if we should skip this URL."""
    m = re.match(r"^https://ourworldindata\.org(/.*)$", url)
    if not m:
        return None
    path = m.group(1).rstrip("/")
    if not path:
        return None

    # /grapher/{slug}
    if path.startswith("/grapher/"):
        slug = path[len("/grapher/"):]
        return ("grapher", slug) if slug and "/" not in slug else None

    # /data-insights/{slug}
    if path.startswith("/data-insights/"):
        slug = path[len("/data-insights/"):]
        return ("data_insight", slug) if slug and "/" not in slug else None

    # Skip noise.
    if path.startswith(("/country/", "/topic/", "/profile/", "/explorers/",
                       "/team/", "/sdgs/")):
        return None

    # Top-level slug == narrative article.
    segs = [s for s in path.split("/") if s]
    if len(segs) == 1 and segs[0] not in SKIP_TOPLEVEL:
        return ("article", segs[0])

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sitemap", default="https://ourworldindata.org/sitemap.xml")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    req = urllib.request.Request(
        args.sitemap,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        xml = r.read().decode()

    urls = re.findall(r"<loc>(https://ourworldindata\.org[^<]+)</loc>", xml)
    print(f"sitemap: {len(urls):,} URLs", flush=True)

    counts = {"grapher": 0, "data_insight": 0, "article": 0, "skip": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for url in urls:
            r = classify(url)
            if r is None:
                counts["skip"] += 1
                continue
            kind, slug = r
            counts[kind] += 1
            f.write(json.dumps({"kind": kind, "slug": slug, "url": url}) + "\n")

    for k, v in counts.items():
        print(f"  {k}: {v:,}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
