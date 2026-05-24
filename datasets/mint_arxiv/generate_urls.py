#!/usr/bin/env python3
"""
Generate download URL list for all arXiv IDs needed by the Nemotron dataset.
Writes urls.txt: one line per PDF: "<url> <local_filename>"
"""

import json
import re
from pathlib import Path

TAR_TO_IDS = Path(__file__).parent / "tar_to_ids.json"
OUT = Path(__file__).parent / "urls.txt"

OLD_ID_RE = re.compile(r"^([a-zA-Z][a-zA-Z.-]*)(\d{4,})$")


def arxiv_url(arxiv_id: str) -> str:
    """Return export.arxiv.org PDF URL for an arXiv ID."""
    if arxiv_id[0].isdigit():
        # New-style: 0704.2978, 1608.03650
        return f"https://export.arxiv.org/pdf/{arxiv_id}"
    else:
        # Old-style: astro-ph0001046 -> astro-ph/0001046
        m = OLD_ID_RE.match(arxiv_id)
        if m:
            return f"https://export.arxiv.org/pdf/{m.group(1)}/{m.group(2)}"
        raise ValueError(f"Cannot parse arXiv ID: {arxiv_id!r}")


def main():
    with open(TAR_TO_IDS) as f:
        tar_to_ids = json.load(f)

    all_ids: set[str] = set()
    for ids in tar_to_ids.values():
        all_ids.update(ids)

    lines = []
    errors = []
    for arxiv_id in sorted(all_ids):
        try:
            url = arxiv_url(arxiv_id)
            # local filename: use arxiv_id as-is (old-style has no slash in id string)
            lines.append(f"{url}\t{arxiv_id}.pdf")
        except ValueError as e:
            errors.append(str(e))

    OUT.write_text("\n".join(lines) + "\n")
    print(f"Written {len(lines)} URLs to {OUT}")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
