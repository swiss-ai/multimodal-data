#!/usr/bin/env python3

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

INPUT_DIR = Path(os.environ.get("FACECAPTION_INPUT_DIR", ""))
OUTPUT_DIR = Path(os.environ.get("FACECAPTION_CROP_DIR", ""))
DEFAULT_WORKER_COUNT = 16

Image.MAX_IMAGE_PIXELS = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-id", required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(DEFAULT_WORKER_COUNT, os.cpu_count() or 1),
    )
    return parser.parse_args()


def add_bytes_to_tar(output_tar, name, data):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = int(time.time())
    output_tar.addfile(info, io.BytesIO(data))


def cleanup_stale_chunk_dirs(part_dir):
    for tmp_dir in sorted(part_dir.glob("chunk_*.tmp")):
        final_dir = part_dir / tmp_dir.stem
        if (tmp_dir / "_SUCCESS").exists() and not final_dir.exists():
            print(f"promote completed tmp {tmp_dir}")
            tmp_dir.rename(final_dir)
            continue

        print(f"remove stale {tmp_dir}")
        shutil.rmtree(tmp_dir)

    for final_dir in sorted(part_dir.glob("chunk_*")):
        if not final_dir.is_dir() or final_dir.suffix == ".tmp":
            continue
        if not (final_dir / "_SUCCESS").exists():
            print(f"remove stale {final_dir}")
            shutil.rmtree(final_dir)


def parse_box(raw_box):
    if raw_box is None:
        raise ValueError("missing box")

    try:
        box = json.loads(raw_box)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid box JSON: {raw_box!r}") from exc

    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"invalid box shape: {raw_box!r}")

    coords = []
    for value in box:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid box value: {raw_box!r}")
        coord = int(value)
        if coord != value:
            raise ValueError(f"non-integer box value: {raw_box!r}")
        coords.append(coord)

    return coords


def crop_image(image_bytes, raw_box):
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        original_width, original_height = image.size
        x1, y1, x2, y2 = parse_box(raw_box)

        x1 = max(0, min(original_width, x1))
        y1 = max(0, min(original_height, y1))
        x2 = max(0, min(original_width, x2))
        y2 = max(0, min(original_height, y2))
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"box {raw_box!r} is outside image bounds {original_width}x{original_height}")

        cropped = image.crop((x1, y1, x2, y2))
        if cropped.mode != "RGB":
            cropped = cropped.convert("RGB")

        output = io.BytesIO()
        cropped.save(output, format="JPEG", quality=100, subsampling=0)
        cropped_width, cropped_height = cropped.size
    return (
        output.getvalue(),
        cropped_width,
        cropped_height,
        original_width,
        original_height,
    )


def mark_crop_failure(row, error_message):
    row["status"] = "failed_to_crop"
    row["error_message"] = error_message
    row["width"] = None
    row["height"] = None
    row["original_width"] = None
    row["original_height"] = None
    row["exif"] = None
    row["sha256"] = None


def build_stats(rows, start_time, end_time):
    status_counts = Counter(row["status"] for row in rows)
    message_counts = Counter()
    for row in rows:
        if row["status"] == "success":
            message_counts["success"] += 1
        else:
            message_counts[row["error_message"] or row["status"]] += 1

    stats = {
        "count": len(rows),
        "successes": status_counts.get("success", 0),
        "duration": end_time - start_time,
        "start_time": start_time,
        "end_time": end_time,
        "status_dict": dict(sorted(message_counts.items())),
    }
    for status, count in sorted(status_counts.items()):
        stats[status] = count
    return stats


def process_shard(input_parquet, input_tar, output_dir):
    start_time = time.time()
    table = pq.read_table(input_parquet)
    schema = table.schema
    if "box" not in schema.names:
        raise RuntimeError(f"{input_parquet} is missing the box column")

    rows = table.to_pylist()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_id = input_parquet.stem
    output_parquet = output_dir / f"{shard_id}.parquet"
    output_tar = output_dir / f"{shard_id}.tar"
    output_stats = output_dir / f"{shard_id}_stats.json"

    cropped_images = {}
    success_rows = [row for row in rows if row["status"] == "success"]

    if input_tar.exists():
        with tarfile.open(input_tar, "r") as input_tar_file:
            members = {member.name: member for member in input_tar_file.getmembers() if member.isfile()}

            for row in success_rows:
                image_name = f"{row['key']}.jpg"
                member = members.get(image_name)
                if member is None:
                    mark_crop_failure(row, f"missing {image_name} in input tar")
                    continue

                input_file = input_tar_file.extractfile(member)
                if input_file is None:
                    mark_crop_failure(row, f"failed to extract {image_name} from input tar")
                    continue

                try:
                    (
                        cropped_bytes,
                        cropped_width,
                        cropped_height,
                        original_width,
                        original_height,
                    ) = crop_image(input_file.read(), row["box"])
                except Exception as exc:  # pylint: disable=broad-except
                    mark_crop_failure(row, str(exc))
                    continue

                row["width"] = cropped_width
                row["height"] = cropped_height
                row["original_width"] = original_width
                row["original_height"] = original_height
                row["error_message"] = None
                row["exif"] = "{}"
                row["sha256"] = hashlib.sha256(cropped_bytes).hexdigest()
                cropped_images[row["key"]] = cropped_bytes
    else:
        for row in success_rows:
            mark_crop_failure(row, f"missing {input_tar.name}")

    tmp_suffix = f".tmp.{os.getpid()}"
    tmp_parquet = output_dir / f"{output_parquet.name}{tmp_suffix}"
    tmp_stats = output_dir / f"{output_stats.name}{tmp_suffix}"
    tmp_tar = output_dir / f"{output_tar.name}{tmp_suffix}"

    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp_parquet)
    tmp_parquet.replace(output_parquet)

    if cropped_images:
        with tarfile.open(tmp_tar, "w") as output_tar_file:
            for row in rows:
                if row["status"] != "success":
                    continue

                key = row["key"]
                add_bytes_to_tar(output_tar_file, f"{key}.jpg", cropped_images[key])
                add_bytes_to_tar(
                    output_tar_file,
                    f"{key}.json",
                    json.dumps(row, indent=4).encode("utf-8"),
                )
                add_bytes_to_tar(
                    output_tar_file,
                    f"{key}.txt",
                    (row["caption"] or "").encode("utf-8"),
                )
        tmp_tar.replace(output_tar)
    else:
        if tmp_tar.exists():
            tmp_tar.unlink()
        if output_tar.exists():
            output_tar.unlink()

    end_time = time.time()
    stats = build_stats(rows, start_time, end_time)
    tmp_stats.write_text(json.dumps(stats, indent=4))
    tmp_stats.replace(output_stats)

    return (
        shard_id,
        len(rows),
        stats["successes"],
        stats.get("failed_to_crop", 0),
    )


def process_chunk(input_chunk_dir, output_part_dir, workers):
    final_dir = output_part_dir / input_chunk_dir.name
    chunk_done_file = final_dir / "_SUCCESS"
    if chunk_done_file.exists():
        print(f"skip {final_dir}")
        return

    tmp_dir = output_part_dir / f"{input_chunk_dir.name}.tmp"
    if final_dir.exists():
        print(f"remove stale {final_dir}")
        shutil.rmtree(final_dir)
    if tmp_dir.exists():
        print(f"remove stale {tmp_dir}")
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    shard_parquets = sorted(input_chunk_dir.glob("*.parquet"))
    tasks = [
        (input_parquet, input_chunk_dir / f"{input_parquet.stem}.tar", tmp_dir) for input_parquet in shard_parquets
    ]

    if workers == 1:
        results = [process_shard(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("fork")) as executor:
            futures = [executor.submit(process_shard, *task) for task in tasks]
            results = [future.result() for future in as_completed(futures)]

    for shard_id, row_count, success_count, failed_to_crop in sorted(results):
        print(
            f"{input_chunk_dir.name}/{shard_id} rows={row_count} "
            f"success={success_count} failed_to_crop={failed_to_crop}"
        )

    (tmp_dir / "_SUCCESS").write_text("")
    tmp_dir.rename(final_dir)
    print(f"done {final_dir}")


def process_part(part_id, workers):
    input_part_dir = INPUT_DIR / part_id
    if not input_part_dir.exists():
        print(f"missing {input_part_dir}, skipping")
        return

    input_done_file = input_part_dir / "_SUCCESS"
    if not input_done_file.exists():
        print(f"incomplete input part {part_id}, skipping")
        return

    part_dir = OUTPUT_DIR / part_id
    done_file = part_dir / "_SUCCESS"
    if done_file.exists():
        print(f"already done {part_id}")
        return

    part_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_chunk_dirs(part_dir)

    input_chunk_dirs = sorted(
        chunk_dir for chunk_dir in input_part_dir.glob("chunk_*") if chunk_dir.is_dir() and chunk_dir.suffix != ".tmp"
    )

    for input_chunk_dir in input_chunk_dirs:
        if not (input_chunk_dir / "_SUCCESS").exists():
            print(f"skip incomplete chunk {input_chunk_dir}")
            continue
        process_chunk(input_chunk_dir, part_dir, workers)

    done_file.write_text("")
    print(f"finished {part_id}")


def main():
    args = parse_args()
    process_part(args.part_id, max(1, args.workers))


if __name__ == "__main__":
    main()
