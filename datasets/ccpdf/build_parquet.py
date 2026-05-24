import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

IMAGES_DIR = Path("/tmp/toolbox/ccpdf/images")
JSONL_DIR = Path(
    "/path/to/data/vision-datasets/raw/sft/nemotron_image_training_v3/hf___nvidia___Nemotron-Image-Training-v3"
)
OUTPUT_DIR = Path(
    "/path/to/data/vision-datasets/raw/sft/nemotron_image_training_v3/swissai___Nemotron-Image-Training-v3"
)

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("messages", pa.string()),
        ("images", pa.map_(pa.string(), pa.binary())),
    ]
)

BATCH_SIZE = 50  # rows per row group, ~460 MB avg, safely under pa.binary() 2.1 GB limit
ROWS_PER_FILE = 4000  # 80 row groups per file
WORKERS = 32


def _load_image(path: str) -> tuple[str, bytes | None]:
    p = IMAGES_DIR / path
    try:
        return path, p.read_bytes()
    except OSError:
        return path, None


def _extract_image_paths(sample: dict) -> list[str]:
    paths = []
    for message in sample.get("messages", []):
        for fragment in message.get("content", []):
            if isinstance(fragment, dict) and fragment.get("type") == "image":
                paths.append(fragment["image"])
    return paths


def process_shard(shard_num: str) -> None:
    shard_name = f"long_document_ccpdf_{shard_num}"
    jsonl_path = JSONL_DIR / shard_name / f"{shard_name}.jsonl"
    shard_dir = OUTPUT_DIR / shard_name
    # Sentinel: presence of part-00000 means shard is complete
    sentinel = shard_dir / f"{shard_name}-00000.parquet"

    if sentinel.exists():
        print(f"Shard {shard_num}: already done, skipping", flush=True)
        return

    shard_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    with open(jsonl_path) as f:
        for line in f:
            samples.append(json.loads(line))

    n_files = (len(samples) + ROWS_PER_FILE - 1) // ROWS_PER_FILE
    print(
        f"Shard {shard_num}: {len(samples)} samples -> {n_files} parquet files",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for file_idx in range(n_files):
            file_samples = samples[file_idx * ROWS_PER_FILE : (file_idx + 1) * ROWS_PER_FILE]
            out_path = shard_dir / f"{shard_name}-{file_idx:05d}.parquet"
            tmp_path = out_path.with_suffix(".parquet.tmp")
            tmp_path.unlink(missing_ok=True)

            total_missing = 0
            with pq.ParquetWriter(tmp_path, SCHEMA) as writer:
                for batch_start in range(0, len(file_samples), BATCH_SIZE):
                    batch = file_samples[batch_start : batch_start + BATCH_SIZE]
                    sample_paths = [_extract_image_paths(s) for s in batch]
                    unique_paths = list({p for paths in sample_paths for p in paths})
                    path_to_bytes = dict(executor.map(_load_image, unique_paths))

                    ids, messages_list, images_list = [], [], []
                    missing = 0
                    for sample, paths in zip(batch, sample_paths):
                        ids.append(sample["id"])
                        messages_list.append(json.dumps(sample["messages"]))
                        img_map = [(p, path_to_bytes[p]) for p in paths if path_to_bytes.get(p) is not None]
                        missing += len(paths) - len(img_map)
                        images_list.append(img_map)

                    table = pa.table(
                        {
                            "id": pa.array(ids, type=pa.string()),
                            "messages": pa.array(messages_list, type=pa.string()),
                            "images": pa.array(images_list, type=pa.map_(pa.string(), pa.binary())),
                        },
                        schema=SCHEMA,
                    )
                    writer.write_table(table)
                    total_missing += missing

            tmp_path.rename(out_path)
            done_rows = file_idx * ROWS_PER_FILE + len(file_samples)
            print(
                f"  [{file_idx + 1}/{n_files}] {out_path.name} ({done_rows}/{len(samples)} rows)"
                + (f" {total_missing} missing" if total_missing else ""),
                flush=True,
            )

    print(f"Shard {shard_num}: done", flush=True)


if __name__ == "__main__":
    shards = sys.argv[1:] if len(sys.argv) > 1 else [f"{i:02d}" for i in range(1, 12)]
    for shard in shards:
        process_shard(shard)
