from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq

DATASET_ROOT = Path("/path/to/data/vision-datasets/hf___common-canvas___commoncatalog-cc-by")
ALLOWED_LICENSE_URLS = {"http://creativecommons.org/licenses/by/2.0/"}
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "commoncatalog"
SYSTEM_PROMPT = (
    "You write natural image captions for multimodal LLM training. Return only the caption text in plain prose."
)
PROMPT = """Write a natural, self-contained caption for this image in 1 to 3 sentences, usually 40 to 90 words.
Keep the prose seamless and human-written, like high-quality dataset text rather than an analysis.
Describe only what is visibly present: the main subjects, their actions, the setting, important spatial relations, visually salient lighting or medium cues, and any clearly legible text.
Be specific and concrete, but do not guess beyond the image.
Do not mention the user, the prompt, the image itself, or your reasoning.
Do not use bullet points, numbered lists, section headers, markdown, XML tags, or meta phrases such as "the user wants", "I need to", or "this image shows".
Start directly with the subject or scene."""


def iter_samples_from_parquet(parquet_path: Path) -> Iterator[dict]:
    relative_path = parquet_path.relative_to(DATASET_ROOT).as_posix()
    parquet_file = pq.ParquetFile(parquet_path)
    row_index = 0
    skipped_for_license = 0

    for row_group_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(
            row_group_index,
            columns=["jpg", "licenseurl", "url"],
        )
        jpg_column = table["jpg"]
        license_column = table["licenseurl"]
        url_column = table["url"]

        for offset in range(table.num_rows):
            license_value = license_column[offset].as_py()
            if license_value not in ALLOWED_LICENSE_URLS:
                skipped_for_license += 1
                continue

            current_row_index = row_index + offset
            sample_id = f"{relative_path}:{current_row_index}"
            yield {
                "sample_id": sample_id,
                "image_bytes": jpg_column[offset].as_py(),
                "media_type": "image/jpeg",
                "metadata": {
                    "path": relative_path,
                    "row_index": current_row_index,
                    "url": url_column[offset].as_py(),
                },
            }

        row_index += table.num_rows

    if skipped_for_license:
        print(f"Skipped {skipped_for_license} rows in {relative_path} due to license filter")


def iter_samples(task_id: int, task_count: int) -> Iterator[dict]:
    parquet_paths = sorted(DATASET_ROOT.rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {DATASET_ROOT}")

    total = len(parquet_paths)
    assigned_paths = parquet_paths[total * task_id // task_count : total * (task_id + 1) // task_count]
    print(
        f"Discovered {len(parquet_paths)} parquet files under {DATASET_ROOT}; "
        f"task {task_id}/{task_count - 1} assigned {len(assigned_paths)}"
    )
    if assigned_paths:
        print(f"Assigned first parquet: {assigned_paths[0]}")
        print(f"Assigned last parquet:  {assigned_paths[-1]}")

    for parquet_path in assigned_paths:
        yield from iter_samples_from_parquet(parquet_path)


loader = SimpleNamespace(
    name="commoncatalog",
    output_dir=OUTPUT_DIR,
    prompt=PROMPT,
    system_prompt=SYSTEM_PROMPT,
    iter_samples=iter_samples,
)
