#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path

import webdataset as wds

DATASET_ROOT = Path("/path/to/data/vision-datasets/hf___Mitsua___art-museums-pd-440k")
CAPTIONS_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "mitsua_art_museums_pd_440k"
DEFAULT_OUTPUT_ROOT = Path("/path/to/data/vision-datasets/hf___Mitsua___art-museums-pd-440k___recap")
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
DEFAULT_INDEX_PATH = DEFAULT_CACHE_DIR / "mitsua_art_museums_pd_440k_captions.sqlite3"
DEFAULT_SAMPLE_DIR = "sample"
SHARD_MAXCOUNT = 10_000
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp")
INSERT_BATCH_SIZE = 1_000
PROGRESS_EVERY = 10_000


def normalize_caption(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def discover_languages() -> list[str]:
    languages = sorted(path.name for path in CAPTIONS_ROOT.iterdir() if path.is_dir() and not path.name.startswith("."))
    if "en" not in languages:
        raise FileNotFoundError(f"Missing English captions under {CAPTIONS_ROOT}")
    return languages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--subdir", default=".")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


def resolve_output_dir(output_root: Path, subdir: str) -> Path:
    if subdir in {"", "."}:
        return output_root
    return output_root / subdir


def iter_source_samples(limit: int | None) -> Iterator[dict]:
    yielded = 0
    shard_paths = sorted(DATASET_ROOT.glob("*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found under {DATASET_ROOT}")

    for shard_path in shard_paths:
        dataset = wds.WebDataset(str(shard_path), shardshuffle=False, empty_check=False)
        for sample in dataset:
            image_key = next((key for key in IMAGE_EXTENSIONS if key in sample), None)
            if image_key is None:
                continue

            metadata = json.loads(bytes(sample.get("json", b"{}")).decode("utf-8"))
            yield {
                "sample_id": sample["__key__"],
                "image_ext": image_key,
                "image_bytes": bytes(sample[image_key]),
                "metadata": {
                    **metadata,
                    "source_shard": shard_path.name,
                    "source_tar": sample.get("__url__", str(shard_path)),
                },
            }
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def ensure_caption_index(
    index_path: Path,
    languages: list[str],
    rebuild: bool,
) -> None:
    if index_path.exists() and not rebuild:
        print(f"Reusing caption index at {index_path}", flush=True)
        return

    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        index_path.unlink()

    print(f"Building caption index at {index_path}", flush=True)
    connection = sqlite3.connect(index_path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(
            """
            CREATE TABLE captions (
                sample_id TEXT NOT NULL,
                lang TEXT NOT NULL,
                caption TEXT NOT NULL,
                PRIMARY KEY (sample_id, lang)
            ) WITHOUT ROWID
            """
        )

        total_inserted = 0
        pending_rows: list[tuple[str, str, str]] = []
        with connection:
            for language in languages:
                language_dir = CAPTIONS_ROOT / language
                for caption_path in sorted(language_dir.glob("captions_task*.jsonl")):
                    with caption_path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            payload = json.loads(line)
                            sample_id = payload.get("sample_id")
                            caption = normalize_caption(payload.get("caption"))
                            if not sample_id or not caption:
                                continue
                            pending_rows.append((sample_id, language, caption))
                            if len(pending_rows) >= INSERT_BATCH_SIZE:
                                connection.executemany(
                                    "INSERT OR REPLACE INTO captions VALUES (?, ?, ?)",
                                    pending_rows,
                                )
                                total_inserted += len(pending_rows)
                                pending_rows.clear()
                                if total_inserted % PROGRESS_EVERY == 0:
                                    print(
                                        f"Indexed {total_inserted} captions so far",
                                        flush=True,
                                    )

            if pending_rows:
                connection.executemany(
                    "INSERT OR REPLACE INTO captions VALUES (?, ?, ?)",
                    pending_rows,
                )
                total_inserted += len(pending_rows)

        print(f"Completed caption index with {total_inserted} rows", flush=True)
    finally:
        connection.close()


def shuffled_languages(sample_id: str, languages: list[str], seed: int) -> list[str]:
    other_languages = [language for language in languages if language != "en"]
    digest = hashlib.sha1(f"{sample_id}:{seed}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], byteorder="big"))
    rng.shuffle(other_languages)
    return ["en", *other_languages]


def format_multilingual_text(
    sample_id: str,
    captions: dict[str, str],
    languages: list[str],
    seed: int,
) -> tuple[str, list[str]]:
    ordered_languages = [
        language for language in shuffled_languages(sample_id, languages, seed) if language in captions
    ]
    sections = [f"lang={language}\n{captions[language]}" for language in ordered_languages]
    return "\n\n".join(sections), ordered_languages


def fetch_captions(connection: sqlite3.Connection, sample_id: str) -> dict[str, str]:
    rows = connection.execute(
        "SELECT lang, caption FROM captions WHERE sample_id = ?",
        (sample_id,),
    ).fetchall()
    return {language: caption for language, caption in rows}


def build_sample_payload(
    source_sample: dict,
    captions: dict[str, str],
    languages: list[str],
    seed: int,
) -> dict[str, bytes]:
    sample_id = source_sample["sample_id"]
    text, ordered_languages = format_multilingual_text(
        sample_id=sample_id,
        captions=captions,
        languages=languages,
        seed=seed,
    )
    metadata = {
        **source_sample["metadata"],
        "sample_id": sample_id,
        "text_template": "lang=<code>\\n<caption> blocks separated by blank lines",
        "language_order": ordered_languages,
    }
    return {
        "__key__": sample_id,
        source_sample["image_ext"]: source_sample["image_bytes"],
        "txt": text.encode("utf-8"),
        "json": json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
    }


def export_samples(
    source_samples: Iterable[dict],
    connection: sqlite3.Connection,
    languages: list[str],
    output_dir: Path,
    seed: int,
    limit: int | None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / "part-%06d.tar")
    written = 0

    with wds.ShardWriter(output_pattern, maxcount=SHARD_MAXCOUNT) as sink:
        for source_sample in source_samples:
            captions = fetch_captions(connection, source_sample["sample_id"])
            if "en" not in captions:
                raise RuntimeError(f"Missing English recap for sample {source_sample['sample_id']!r}")

            sink.write(
                build_sample_payload(
                    source_sample=source_sample,
                    captions=captions,
                    languages=languages,
                    seed=seed,
                )
            )
            written += 1
            if written % PROGRESS_EVERY == 0:
                if limit is None:
                    print(f"Wrote {written} museum samples", flush=True)
                else:
                    print(f"Wrote {written}/{limit} museum samples", flush=True)

    return written


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_root, args.subdir)
    languages = discover_languages()

    ensure_caption_index(
        index_path=args.index_path,
        languages=languages,
        rebuild=args.rebuild_index,
    )

    read_connection = sqlite3.connect(args.index_path)
    try:
        written = export_samples(
            source_samples=iter_source_samples(args.limit),
            connection=read_connection,
            languages=languages,
            output_dir=output_dir,
            seed=args.seed,
            limit=args.limit,
        )
    finally:
        read_connection.close()

    summary = {
        "index_path": str(args.index_path),
        "languages": languages,
        "limit": args.limit,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "text_template": "lang=<code>\\n<caption> blocks separated by blank lines",
        "written": written,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
