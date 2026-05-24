import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import webdataset as wds

DATASET_ROOT = Path("/path/to/data/vision-datasets/hf___Mitsua___art-museums-pd-440k")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "mitsua_art_museums_pd_440k" / "en"
SYSTEM_PROMPT = (
    "You write detailed museum descriptions for multimodal LLM training. "
    "Use the image as primary evidence, use the provided source captions only as "
    "supporting hints, and return only the final English description in plain prose."
)
PROMPT_TEMPLATE = """Write an extensive English description for this museum image in 2 to 4 sentences, usually 90 to 160 words.
Use the image as the primary source of truth.
Use the English and Japanese source captions only as supporting hints for object type, material, technique, title, or subject when they are consistent with the image.
Transform terse keywords into fluent prose; do not copy the source captions or keep comma-separated lists.
Focus on museum-relevant visible details: the main object or scene, composition, materials, colors, textures, ornament, craftsmanship, pose or iconography, and any visible signs of age, wear, or display context.
If the source captions strongly suggest a specific object type, medium, or subject and it fits the image, weave that detail in naturally.
Do not invent dates, provenance, symbolism, artist biography, or details unsupported by the image and source captions.
Do not mention the prompt, the source captions, the Japanese language, uncertainty, or your reasoning.
Return only the final English description.

English source caption:
{caption_en}

Japanese source caption:
{caption_jp}
"""
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


def normalize_caption(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def build_prompt(sample: dict) -> str:
    metadata = sample.get("metadata") or {}
    caption_en = normalize_caption(metadata.get("captionEn"))
    caption_jp = normalize_caption(metadata.get("captionJp"))
    if not caption_en and not caption_jp:
        raise ValueError(f"Sample {sample['sample_id']!r} is missing both captions")

    return PROMPT_TEMPLATE.format(
        caption_en=caption_en or "[missing]",
        caption_jp=caption_jp or "[missing]",
    )


def iter_samples_from_shard(shard_path: Path) -> Iterator[dict]:
    dataset = wds.WebDataset(str(shard_path), shardshuffle=False, empty_check=False)

    for sample in dataset:
        image_bytes, media_type = extract_image(sample)
        raw_metadata = sample.get("json", b"{}")
        metadata = json.loads(bytes(raw_metadata).decode("utf-8"))
        sample_id = sample["__key__"]
        metadata_key = metadata.get("ImageID")
        if metadata_key is not None and metadata_key != sample_id:
            raise ValueError(f"Mismatched ImageID for {shard_path}: sample={sample_id!r} metadata={metadata_key!r}")

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
    name="mitsua_art_museums_pd_440k___en",
    output_dir=OUTPUT_DIR,
    system_prompt=SYSTEM_PROMPT,
    build_prompt=build_prompt,
    iter_samples=iter_samples,
)
