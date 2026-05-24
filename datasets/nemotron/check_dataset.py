#!/usr/bin/env python3
"""Check a single dataset's mirrored parquet version against the original JSONL."""

import argparse
import sys
from pathlib import Path

import orjson as json
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ORIG_BASE = Path(
    "/path/to/data/vision-datasets/raw/sft/nemotron_image_training_v3/hf___nvidia___Nemotron-Image-Training-v3"
)
MIRROR_BASE = Path(
    "/path/to/data/vision-datasets/raw/sft/nemotron_image_training_v3/swissai___Nemotron-Image-Training-v3"
)

EXPECTED_FIELD_NAMES = ["id", "messages", "images"]


def count_images_in_messages(messages):
    count = 0
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    count += 1
    return count


def read_original(dataset_name):
    """Return {id: n_images} from original JSONL."""
    jsonl_path = ORIG_BASE / dataset_name / f"{dataset_name}.jsonl"
    if not jsonl_path.exists():
        print(f"[{dataset_name}] Original JSONL not found: {jsonl_path}", file=sys.stderr)
        return {}

    id_to_nimages = {}
    with open(jsonl_path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            id_to_nimages[obj["id"]] = count_images_in_messages(obj["messages"])
    return id_to_nimages


def _image_refs_from_messages(messages_json: str) -> set[str]:
    """Extract the set of image path strings referenced in a messages JSON blob."""
    msgs = json.loads(messages_json)
    return {
        item["image"]
        for msg in msgs
        for item in (msg.get("content") or [])
        if isinstance(item, dict) and item.get("type") == "image"
    }


def check_references_integrity(dataset_name) -> int:
    """Return 1 if every image ref in messages exists as a key in the images map.

    Samples the first row group of the first parquet file.  Systematic path
    problems (wrong subdirectory prefix, wrong filename) show up in every row,
    so one row group is sufficient to catch them without loading whole files.
    Returns 1 on pass, 0 on any mismatch, and 1 if the dataset cannot be read
    (deferred to valid_structure / valid_samples to report that problem).
    """
    mirror_dir = MIRROR_BASE / dataset_name
    if not mirror_dir.exists():
        return 1

    parquet_files = sorted(mirror_dir.glob("*.parquet"))
    if not parquet_files:
        return 1

    try:
        pf = pq.ParquetFile(parquet_files[0])
        if pf.metadata.num_row_groups == 0:
            return 1
        rg = pf.read_row_group(0)
    except Exception:
        return 1  # unreadable file; structural problem reported elsewhere

    for i in range(rg.num_rows):
        refs = _image_refs_from_messages(rg["messages"][i].as_py())
        map_scalar = rg["images"][i].as_py()
        keys = {k for k, _ in map_scalar} if map_scalar else set()
        if refs != keys:
            return 0

    return 1


def check_schema(parquet_files):
    """Return True if all parquet files match the expected schema."""
    for pf_path in parquet_files:
        schema = pq.read_schema(pf_path)
        if schema.names != EXPECTED_FIELD_NAMES:
            return False
        if schema.field("id").type != pa.string():
            return False
        if schema.field("messages").type != pa.string():
            return False
        images_type = schema.field("images").type
        if not pa.types.is_map(images_type):
            return False
        if images_type.key_type != pa.string():
            return False
        # accept both binary and large_binary
        if images_type.item_type not in (pa.binary(), pa.large_binary()):
            return False
    return True


def read_mirrored(dataset_name):
    """Return (valid_structure, {id: n_images}) from mirrored parquet files."""
    mirror_dir = MIRROR_BASE / dataset_name
    if not mirror_dir.exists():
        return False, {}

    parquet_files = sorted(mirror_dir.glob("*.parquet"))
    if not parquet_files:
        return False, {}

    valid_structure = check_schema(parquet_files)

    id_to_nimages = {}
    for pf_path in parquet_files:
        # Read only id and messages — avoids loading image bytes (can be 50+GB per file).
        # Count image references from the messages JSON string using arrow string ops.
        table = pq.read_table(pf_path, columns=["id", "messages"])
        ids = table.column("id").to_pylist()
        counts = pc.count_substring(table.column("messages"), '"type": "image"').to_pylist()
        for sample_id, n in zip(ids, counts):
            id_to_nimages[sample_id] = n

    return valid_structure, id_to_nimages


def check_dataset(dataset_name, log_dir=None):
    """Check one dataset and return (valid_structure, valid_samples, refs_integrity, n_orig, n_mirror)."""
    orig = read_original(dataset_name)
    valid_structure, mirror = read_mirrored(dataset_name)

    n_orig = len(orig)
    n_mirror = len(mirror)

    mismatches = []
    for sample_id, n_mirror_imgs in mirror.items():
        n_orig_imgs = orig.get(sample_id)
        if n_orig_imgs is None or n_orig_imgs != n_mirror_imgs:
            mismatches.append(sample_id)

    valid_samples = 1 if not mismatches else 0

    if mismatches:
        mismatch_text = "samples with non-matching number of images:\n" + "\n".join(mismatches)
        if log_dir:
            log_path = Path(log_dir) / f"{dataset_name}.log"
            log_path.write_text(mismatch_text + "\n")
        else:
            print(f"[{dataset_name}] {mismatch_text}", file=sys.stderr)

    refs_integrity = check_references_integrity(dataset_name)
    return int(valid_structure), valid_samples, refs_integrity, n_orig, n_mirror


def main():
    parser = argparse.ArgumentParser(description="Check a single mirrored dataset.")
    parser.add_argument("dataset_name", help="Name of the dataset subdirectory")
    parser.add_argument("--log-dir", help="Directory to write per-dataset mismatch logs")
    args = parser.parse_args()

    valid_structure, valid_samples, refs_integrity, n_orig, n_mirror = check_dataset(
        args.dataset_name, log_dir=args.log_dir
    )
    pct = (n_mirror / n_orig) if n_orig > 0 else 0.0
    n_missing = n_orig - n_mirror

    print(
        f"{args.dataset_name}\t{valid_structure}\t{valid_samples}\t{refs_integrity}\t{n_orig}\t{n_mirror}\t{n_missing}\t{pct:.4f}"
    )


if __name__ == "__main__":
    main()
