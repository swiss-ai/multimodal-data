"""Normalize external URLs in WebSight v0.2 HTML to prevent memorization.

Replaces real URLs with generic placeholders while preserving HTML structure:
  - Image src (unsplash, placeholder.com) → https://placeholder.com/{WxH}
  - CDN links (jsdelivr) → kept as-is (Tailwind CSS ref is part of the code pattern)
  - Social media links → https://example.com/social
  - Video embeds (youtube) → https://example.com/video
  - SVG namespaces (w3.org) → kept as-is (required for valid SVG)
  - Fake domains (fashionbrand.com etc.) → kept as-is (already generic)

Usage:
    python normalize_urls.py \
        --input-dir /capstor/.../HuggingFaceM4___WebSight/v0.2 \
        --output-dir /capstor/.../HuggingFaceM4___WebSight/v0.2_normalized
"""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa


# Patterns to normalize (order matters — more specific first)
NORMALIZE_RULES = [
    # Unsplash random images: extract dimensions from URL
    (
        re.compile(r'https?://source\.unsplash\.com/random/(\d+)x(\d+)/[^\s"\'<>)\]]*'),
        lambda m: f"https://placeholder.com/{m.group(1)}x{m.group(2)}",
    ),
    # Unsplash without dimensions
    (
        re.compile(r'https?://source\.unsplash\.com/[^\s"\'<>)\]]*'),
        lambda m: "https://placeholder.com/300x200",
    ),
    # via.placeholder.com — already generic but normalize domain
    (
        re.compile(r'https?://via\.placeholder\.com/(\d+x?\d*)[^\s"\'<>)\]]*'),
        lambda m: f"https://placeholder.com/{m.group(1)}",
    ),
    # YouTube embeds
    (
        re.compile(r'https?://(?:www\.)?youtube\.com/[^\s"\'<>)\]]*'),
        lambda m: "https://example.com/video",
    ),
    # Social media links
    (
        re.compile(r'https?://(?:www\.)?(?:twitter|facebook|instagram)\.com/[^\s"\'<>)\]]*'),
        lambda m: "https://example.com/social",
    ),
    # tailwindui.com links
    (
        re.compile(r'https?://(?:www\.)?tailwindui\.com/[^\s"\'<>)\]]*'),
        lambda m: "https://example.com/ui",
    ),
]

# These domains are kept as-is
KEEP_DOMAINS = {"cdn.jsdelivr.net", "www.w3.org"}


def normalize_html(html: str) -> str:
    for pattern, replacement in NORMALIZE_RULES:
        html = pattern.sub(replacement, html)
    return html


def process_parquet(args: tuple[Path, Path, Path]) -> str:
    pq_path, input_dir, output_dir = args
    rel = pq_path.relative_to(input_dir)
    out_path = output_dir / rel

    table = pq.read_table(pq_path)
    texts = table.column("text").to_pylist()
    normalized = [normalize_html(t) for t in texts]

    idx = table.schema.get_field_index("text")
    new_table = table.set_column(idx, "text", pa.array(normalized, type=pa.string()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, out_path)
    return str(rel)


def main():
    parser = argparse.ArgumentParser(description="Normalize URLs in WebSight HTML")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    parquets = sorted(args.input_dir.rglob("*.parquet"))
    print(f"Found {len(parquets)} parquet files in {args.input_dir}")
    print(f"Using {args.workers} workers")

    tasks = [(pq_path, args.input_dir, args.output_dir) for pq_path in parquets]
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_parquet, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            rel = future.result()
            if done % 50 == 0 or done == len(parquets):
                print(f"  [{done}/{len(parquets)}] {rel}")

    print("Done.")


if __name__ == "__main__":
    main()
