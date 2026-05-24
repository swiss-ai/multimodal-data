#!/usr/bin/env python3
"""Search for prompt leak phrases across webdataset tar files."""

import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PHRASES = [
    "Sentence 1",
    "Sentence 2",
    "proper sentence structure",
    "structure and grammar",
    "well-structured sentence",
    "detailed real-world",
]

DATASETS = {
    "SkyScript": os.environ.get("SKYSCRIPT_PATH", "/path/to/data/vision-datasets/processed/gh___wangzhecheng___SkyScript___processed"),
    "GeoChat": os.environ.get("GEOCHAT_PATH", "/path/to/data/vision-datasets/processed/hf___MBZUAI___GeoChat_Instruct___processed"),
    "MapTrace": os.environ.get("MAPTRACE_PATH", "/path/to/data/vision-datasets/processed/hf___google___MapTrace___processed"),
    "swisstopo": os.environ.get("SWISSTOPO_PATH", "/path/to/data/vision-datasets/processed/swisstopo___map_sat___processed"),
    "ign_city": os.environ.get("IGN_CITY_PATH", "/path/to/data/vision-datasets/processed/ign___ign_city_tiles___processed"),
    "RSTeller": os.environ.get("RSTELLER_PATH", "/path/to/data/vision-datasets/processed/hf___SlytherinGe___RSTeller___processed"),
    "FLAIR-HUB": os.environ.get("FLAIR_HUB_PATH", "/path/to/data/vision-datasets/processed/ign___IGNF___FLAIR-HUB___processed"),
}


def search_tar(args):
    dataset_name, tar_path = args
    hits = []
    try:
        with tarfile.open(tar_path, "r") as tf:
            for member in tf.getmembers():
                if not member.name.endswith(".txt"):
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                try:
                    text = f.read().decode("utf-8", errors="replace")
                except Exception:
                    continue
                for phrase in PHRASES:
                    if phrase.lower() in text.lower():
                        # grab a snippet around the hit
                        idx = text.lower().find(phrase.lower())
                        snippet = text[max(0, idx - 30) : idx + 120].replace("\n", " ")
                        hits.append(
                            {
                                "dataset": dataset_name,
                                "tar": os.path.basename(tar_path),
                                "file": member.name,
                                "phrase": phrase,
                                "snippet": snippet,
                            }
                        )
                        break  # one hit per file is enough
    except Exception as e:
        hits.append({"dataset": dataset_name, "tar": tar_path, "error": str(e)})
    return hits


def main():
    tasks = []
    for name, path in DATASETS.items():
        tars = sorted(Path(path).glob("*.tar"))
        print(f"{name}: {len(tars)} tar files", flush=True)
        for t in tars:
            tasks.append((name, str(t)))

    WORKERS = 250
    print(f"\nSearching {len(tasks)} tar files with {WORKERS} workers...\n", flush=True)

    all_hits = []
    done = 0
    with ProcessPoolExecutor(max_workers=250) as executor:
        futs = {executor.submit(search_tar, t): t for t in tasks}
        for fut in as_completed(futs):
            done += 1
            hits = fut.result()
            if hits and "error" not in hits[0]:
                for h in hits:
                    all_hits.append(h)
                    print(
                        f"[HIT] dataset={h['dataset']}  tar={h['tar']}  file={h['file']}",
                        flush=True,
                    )
                    print(f"      phrase: {h['phrase']!r}", flush=True)
                    print(f"      snippet: {h['snippet']!r}", flush=True)
                    print(flush=True)
            if done % 50 == 0:
                print(f"  ... {done}/{len(tasks)} tars searched", flush=True)

    print(f"\n{'=' * 60}")
    print(f"Done. {len(all_hits)} hits found across {len(tasks)} tar files.")
    if all_hits:
        from collections import Counter

        by_dataset = Counter(h["dataset"] for h in all_hits)
        print("\nHits by dataset:")
        for ds, count in by_dataset.most_common():
            print(f"  {ds}: {count} files")


if __name__ == "__main__":
    main()
