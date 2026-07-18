#!/usr/bin/env python
"""Generate README.md from summary.yaml (the source of truth).

Edit summary.yaml, then run:  python datasets/generate_readme.py
"""
import os
import yaml

DIR = os.path.dirname(os.path.abspath(__file__))
YAML = os.path.join(DIR, "summary.yaml")
README = os.path.join(DIR, "README.md")


def link(path):
    """Markdown link whose text is the last segment of the path."""
    text = path.rstrip("/").split("/")[-1]
    return f"[{text}]({path})"


def paths_md(paths):
    return ", ".join(link(p) for p in paths) if paths else "-"


def upstream_md(u):
    if not u:
        return ""
    url = u.split(" ", 1)[0]
    note = u[len(url):].strip()
    return f"[source]({url})" + (f" {note}" if note else "")


def comment_md(rec):
    fl = ", ".join(link(p) for p in rec.get("filtering", []))
    parts = [x for x in (fl, rec.get("comment", "")) if x]
    return " - ".join(parts)


def main():
    recs = yaml.safe_load(open(YAML))
    cols = ["Dataset", "License", "Modality", "Stage", "Processing", "Upstream", "Comment"]
    out = [
        "# Datasets",
        "",
        "Source of truth: [`summary.yaml`](summary.yaml). This table is generated - edit the YAML, then run `python datasets/generate_readme.py`.",
        "",
        "`Comment` documents license/subset filtering (which Mixed/NC/SA parts were removed or which permissive subset was kept): a linked path points to the filtering code, `TODO` marks filtering still to be documented.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in recs:
        row = [
            r["dataset"],
            r.get("license", "") or "",
            r["modality"],
            ", ".join(r.get("stage", [])),
            paths_md(r.get("processing", [])),
            upstream_md(r.get("upstream", "")),
            comment_md(r),
        ]
        out.append("| " + " | ".join(row) + " |")
    open(README, "w").write("\n".join(out) + "\n")
    print(f"wrote {README} ({len(recs)} rows)")


if __name__ == "__main__":
    main()
