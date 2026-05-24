import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import webdataset as wds

DATASET_ROOT = Path("/path/to/data/vision-datasets/openimages_v7_dense___full")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "openimages_v7_dense___full"
SYSTEM_PROMPT = "You are an expert image analyst. Return only a single cohesive paragraph."
PROMPT = """You are an expert image analyst. Your task is to provide a highly detailed, comprehensive, and accurate description of the provided image.
Follow these strict guidelines:
- Directness: Do not use introductory filler phrases like "This image shows", "A picture of", or "Here we can see". Start describing immediately.
- Main Subject: Describe the main subject(s) in extreme detail, including their appearance, clothing, colors, and actions.
- Composition: Explicitly state the spatial relationships between objects, such as "behind", "on top of", and "to the left".
- Environment: Describe the background, setting, and any secondary objects.
- Atmosphere & Style: Mention the lighting, the mood, and the medium or style, such as photograph, digital art, oil painting, or 3D render.
- Text: If there is any readable text in the image, quote it exactly.
Output one or more cohesive paragraphs that seamlessly integrate all these details.
Make the description really long while staying faithful to what is visibly present.
Do not guess beyond the image.
Do not mention the prompt, the user, your reasoning, or meta phrases like "this image shows"."""
IMAGE_KEYS_TO_MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}


def iter_shard_paths() -> list[Path]:
    shard_paths = sorted(DATASET_ROOT.glob("*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"No WebDataset shards found under {DATASET_ROOT}")
    return shard_paths


def extract_image(sample: dict) -> tuple[bytes, str]:
    for key, media_type in IMAGE_KEYS_TO_MEDIA_TYPES.items():
        image_bytes = sample.get(key)
        if image_bytes is not None:
            return bytes(image_bytes), media_type

    available_keys = ", ".join(sorted(sample))
    raise KeyError(f"No supported image payload found in sample keys: {available_keys}")


def iter_samples_from_shard(shard_path: Path) -> Iterator[dict]:
    dataset = wds.WebDataset(str(shard_path), shardshuffle=False, empty_check=False)

    for sample in dataset:
        image_bytes, media_type = extract_image(sample)
        raw_metadata = sample.get("json", b"{}")
        metadata = json.loads(bytes(raw_metadata).decode("utf-8"))
        sample_id = sample["__key__"]
        metadata_key = metadata.get("__key__")
        if metadata_key is not None and metadata_key != sample_id:
            raise ValueError(f"Mismatched __key__ for {shard_path}: sample={sample_id!r} metadata={metadata_key!r}")

        yield {
            "sample_id": sample_id,
            "image_bytes": image_bytes,
            "media_type": media_type,
            "metadata": {
                **metadata,
                "shard": shard_path.name,
                "source_tar": sample.get("__url__", str(shard_path)),
            },
        }


def iter_samples(task_id: int, task_count: int) -> Iterator[dict]:
    shard_paths = iter_shard_paths()
    total = len(shard_paths)
    assigned_paths = shard_paths[total * task_id // task_count : total * (task_id + 1) // task_count]
    print(
        f"Discovered {len(shard_paths)} shards under {DATASET_ROOT}; "
        f"task {task_id}/{task_count - 1} assigned {len(assigned_paths)}"
    )
    if assigned_paths:
        print(f"Assigned first shard: {assigned_paths[0]}")
        print(f"Assigned last shard:  {assigned_paths[-1]}")

    for shard_path in assigned_paths:
        yield from iter_samples_from_shard(shard_path)


loader = SimpleNamespace(
    name="openimages_v7_dense___full",
    output_dir=OUTPUT_DIR,
    prompt=PROMPT,
    system_prompt=SYSTEM_PROMPT,
    iter_samples=iter_samples,
)
