#!/usr/bin/env python3
"""
Generate download URL list for arXiv IDs.
Writes urls.txt: one line per PDF: "<url> <local_filename>"

Input: JSON file mapping tar_name -> list of arxiv_ids
Output: urls.txt with download URLs and filenames
"""

import argparse
import json
import re
from pathlib import Path

OLD_ID_RE = re.compile(r"^([a-zA-Z][a-zA-Z.-]*)(\d{4,})$")


def arxiv_url(arxiv_id: str) -> str:
    """Return export.arxiv.org PDF URL for an arXiv ID."""
    if arxiv_id[0].isdigit():
        return f"https://export.arxiv.org/pdf/{arxiv_id}"
    else:
        m = OLD_ID_RE.match(arxiv_id)
        if m:
            return f"https://export.arxiv.org/pdf/{m.group(1)}/{m.group(2)}"
        raise ValueError(f"Cannot parse arXiv ID: {arxiv_id!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="tar_to_ids.json",
        help="Input JSON file mapping tar name -> list of arxiv IDs",
    )
    parser.add_argument("-o", "--output", type=str, default="urls.txt", help="Output urls.txt path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path) as f:
        tar_to_ids = json.load(f)

    all_ids: set[str] = set()
    for ids in tar_to_ids.values():
        all_ids.update(ids)

    lines = []
    errors = []
    for arxiv_id in sorted(all_ids):
        try:
            url = arxiv_url(arxiv_id)
            lines.append(f"{url}\t{arxiv_id}.pdf")
        except ValueError as e:
            errors.append(str(e))

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Written {len(lines)} URLs to {output_path}")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
