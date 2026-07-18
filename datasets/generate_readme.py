#!/usr/bin/env python
import os
import yaml

DIR = os.path.dirname(os.path.abspath(__file__))
COLS = [
    "Dataset",
    "License",
    "Modality",
    "Stage",
    "Processing",
    "Upstream",
    "Comment",
]


def links(paths):
    return ", ".join(f"[{p.rstrip('/').split('/')[-1]}]({p})" for p in paths) or "-"


def source(u):
    url, _, note = u.partition(" ")
    return f"[link]({url}) {note}".rstrip() if url else ""


def comment(license_filtering, comment):
    out = ""
    if license_filtering and license_filtering != "-":
        out += "[" + license_filtering + "] "
    out += comment
    return out


def row(r):
    return [
        r["dataset"],
        r["license"],
        r["modality"],
        ", ".join(r["stage"]),
        links(r["processing"]),
        source(r["upstream"]),
        comment(
            links(r["license_filtering"]),
            r["comment"],
        ),
    ]


def main():
    recs = yaml.safe_load(open(os.path.join(DIR, "summary.yaml")))
    lines = [
        "# Datasets",
        "",
        "Auto-generated from [`summary.yaml`](summary.yaml) with `python datasets/generate_readme.py`.",
        "",
        "| " + " | ".join(COLS) + " |",
        "| " + " | ".join(["---"] * len(COLS)) + " |",
        *["| " + " | ".join(row(r)) + " |" for r in recs],
    ]
    open(os.path.join(DIR, "README.md"), "w").write("\n".join(lines) + "\n")
    print(f"wrote README.md ({len(recs)} rows)")


if __name__ == "__main__":
    main()
