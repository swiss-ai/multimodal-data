import json
import warnings
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import webdataset as wds
from loader_openimages_v7_dense___full_v2 import PROMPT, SYSTEM_PROMPT
from PIL import Image, ImageFile
from vllm import SamplingParams

DATASET_ROOT = Path("/path/to/data/vision-datasets/hf___Spawning___pd12m-full")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "spawning_pd12m_full_v2"
IMAGE_KEYS_TO_MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}
SAVE_FORMAT_TO_MEDIA_TYPE = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


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


def sanitize_image_if_needed(
    image_bytes: bytes,
    media_type: str,
    sample_id: str,
) -> tuple[bytes, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
                image.getexif()
                return image_bytes, media_type
        except Exception as exc:
            print(f"Sanitizing {sample_id}: {type(exc).__name__}: {exc}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            image_format = (image.format or "").upper()
            save_format = image_format if image_format in SAVE_FORMAT_TO_MEDIA_TYPE else "PNG"
            clean_image = image.copy()

            if save_format == "JPEG" and clean_image.mode not in ("L", "RGB"):
                clean_image = clean_image.convert("RGB")
            elif save_format == "WEBP" and clean_image.mode not in ("L", "RGB", "RGBA"):
                clean_image = clean_image.convert("RGB")

            encoded = BytesIO()
            save_kwargs = {}
            if save_format == "JPEG":
                save_kwargs["quality"] = 95
            elif save_format == "WEBP":
                save_kwargs["quality"] = 95

            clean_image.save(encoded, format=save_format, **save_kwargs)
            return encoded.getvalue(), SAVE_FORMAT_TO_MEDIA_TYPE[save_format]


def iter_raw_samples_from_shard(shard_path: Path) -> Iterator[dict]:
    dataset = wds.WebDataset(str(shard_path), shardshuffle=False, empty_check=False)
    yield from dataset


def build_sample_record(sample: dict, shard_path: Path) -> dict:
    image_bytes, media_type = extract_image(sample)
    raw_metadata = sample.get("json", b"{}")
    metadata = json.loads(bytes(raw_metadata).decode("utf-8"))
    sample_id = sample["__key__"]
    metadata_key = metadata.get("key")
    if metadata_key is not None and str(metadata_key) != sample_id:
        raise ValueError(f"Mismatched key for {shard_path}: sample={sample_id!r} metadata={metadata_key!r}")
    image_bytes, media_type = sanitize_image_if_needed(
        image_bytes,
        media_type,
        sample_id,
    )

    return {
        "sample_id": sample_id,
        "image_bytes": image_bytes,
        "media_type": media_type,
        "metadata": {
            **metadata,
            "shard": shard_path.name,
            "source_tar": sample.get("__url__", str(shard_path)),
        },
    }


def iter_samples_from_shard(shard_path: Path) -> Iterator[dict]:
    for sample in iter_raw_samples_from_shard(shard_path):
        yield build_sample_record(sample, shard_path)


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


def iter_samples_resume(
    task_id: int,
    task_count: int,
    *,
    processed_count: int,
    processed_sample_ids: set[str] | None = None,
) -> Iterator[dict]:
    del processed_sample_ids

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

    remaining_skip = processed_count
    if remaining_skip:
        print(f"Fast resume enabled for {task_id}: skipping {remaining_skip} samples before image sanitation")

    for shard_path in assigned_paths:
        for sample in iter_raw_samples_from_shard(shard_path):
            if remaining_skip > 0:
                remaining_skip -= 1
                continue
            yield build_sample_record(sample, shard_path)


loader = SimpleNamespace(
    name="spawning_pd12m_full_v2",
    output_dir=OUTPUT_DIR,
    prompt=PROMPT,
    system_prompt=SYSTEM_PROMPT,
    iter_samples=iter_samples,
    iter_samples_resume=iter_samples_resume,
    batch_size=64,
    sampling_params=SamplingParams(
        temperature=0.3,
        top_p=0.7,
        top_k=20,
        repetition_penalty=1.15,
        max_tokens=768,
    ),
)
