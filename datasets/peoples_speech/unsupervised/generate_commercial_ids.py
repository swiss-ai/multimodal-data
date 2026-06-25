#!/usr/bin/env python3
"""
Generate list of archive IDs with commercially-usable licenses.

Parses licenses.jsonl from the HF hub and extracts archive IDs whose license
allows commercial use: Public Domain, CC0, US Government Works, CC-BY.
Excludes: CC-BY-SA, CC-BY-NC, and any other restrictive licenses.
"""

import argparse
import json


def is_commercial_use_allowed(url):
    """Return True if license allows commercial use without SA/NC restrictions."""
    url = url.lower().strip()

    # Public domain variants
    if "publicdomain" in url or "public_domain" in url:
        return True
    if "/cc0/" in url or "/zero/" in url:
        return True

    # US Government works
    if "usa.gov/government-works" in url:
        return True

    # CC-BY (without SA or NC)
    if "/by/" in url or "/by-" in url:
        if "-sa" in url or "-nc" in url:
            return False
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Generate commercial-use archive IDs")
    parser.add_argument("--licenses-jsonl", required=True, help="Path to licenses.jsonl")
    parser.add_argument("--output", required=True, help="Output path for IDs file")
    args = parser.parse_args()

    included = []
    excluded_counts = {}
    included_counts = {}
    total = 0

    with open(args.licenses_jsonl) as f:
        for line in f:
            total += 1
            record = json.loads(line)
            metadata = record.get("metadata", {})
            lic = metadata.get("licenseurl", "NONE") if isinstance(metadata, dict) else "NONE"
            if isinstance(lic, list):
                lic = lic[0] if lic else "NONE"
            lic = str(lic)
            identifier = record.get("identifier", "")

            if is_commercial_use_allowed(lic):
                included.append(identifier)
                included_counts[lic] = included_counts.get(lic, 0) + 1
            else:
                excluded_counts[lic] = excluded_counts.get(lic, 0) + 1

    # Write IDs
    with open(args.output, "w") as f:
        for aid in sorted(included):
            f.write(aid + "\n")

    # Summary
    print(f"Total archives:    {total}")
    print(f"Included:          {len(included)} ({100*len(included)/total:.1f}%)")
    print(f"Excluded:          {total - len(included)} ({100*(total-len(included))/total:.1f}%)")

    print(f"\nIncluded licenses (top 10):")
    for lic, count in sorted(included_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:>7d}  {lic}")

    print(f"\nExcluded licenses (top 10):")
    for lic, count in sorted(excluded_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:>7d}  {lic}")

    print(f"\nSaved {len(included)} IDs to: {args.output}")


if __name__ == "__main__":
    main()
