import json
import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import webdataset as wds

DATASET_ROOT = Path("/path/to/data/vision-datasets/hf___Mitsua___art-museums-pd-440k")
ENGLISH_CAPTIONS_DIR = Path(__file__).resolve().parent / "outputs" / "mitsua_art_museums_pd_440k" / "en"
IMAGE_KEYS_TO_MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}
LANGUAGE_CONFIGS = {
    "fr": {"name": "French", "output_subdir": "fr"},
    "pt": {"name": "Portuguese", "output_subdir": "pt"},
    "de": {"name": "German", "output_subdir": "de"},
    "it": {"name": "Italian", "output_subdir": "it"},
    "ru": {"name": "Russian", "output_subdir": "ru"},
    "zh-hans": {"name": "Simplified Chinese", "output_subdir": "zh-hans"},
    "zh-hant": {"name": "Traditional Chinese", "output_subdir": "zh-hant"},
    "ja": {"name": "Japanese", "output_subdir": "ja"},
    "ko": {"name": "Korean", "output_subdir": "ko"},
    "ar": {"name": "Arabic", "output_subdir": "ar"},
    "hi": {"name": "Hindi", "output_subdir": "hi"},
    "tr": {"name": "Turkish", "output_subdir": "tr"},
    "vi": {"name": "Vietnamese", "output_subdir": "vi"},
    "id": {"name": "Indonesian", "output_subdir": "id"},
    "th": {"name": "Thai", "output_subdir": "th"},
}
TARGET_LANG = os.environ["RECAPTION_TARGET_LANG"]
try:
    TARGET_CONFIG = LANGUAGE_CONFIGS[TARGET_LANG]
except KeyError as exc:
    raise KeyError(
        f"Unsupported RECAPTION_TARGET_LANG={TARGET_LANG!r}; expected one of {sorted(LANGUAGE_CONFIGS)}"
    ) from exc

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "mitsua_art_museums_pd_440k" / TARGET_CONFIG["output_subdir"]
SYSTEM_PROMPT = (
    "You write detailed museum descriptions for multimodal LLM training. "
    "Use the image as primary evidence, use the provided English museum "
    "description only as supporting context, and return only the final "
    f"{TARGET_CONFIG['name']} description in plain prose."
)
PROMPT_TEMPLATE = """Write an extensive description in {language_name} for this museum image in 2 to 4 sentences.
Use the image as the primary source of truth.
Use the English museum description only as a supporting hint for object type, material, technique, title, subject, or composition when it matches the image.
Write natural, idiomatic {language_name} prose rather than a literal word-for-word translation.
Keep the detail level high and museum-relevant: describe the main object or scene, composition, materials, colors, textures, ornament, craftsmanship, pose or iconography, and any visible signs of age, wear, or display context.
If the English description strongly suggests a specific object type, medium, or subject and it fits the image, weave that detail in naturally.
Do not invent dates, provenance, symbolism, artist biography, or details unsupported by the image and English description.
Do not mention the English description, the prompt, uncertainty, or your reasoning.
Return only the final description in {language_name}.

English museum description:
{caption_en}
"""
_ENGLISH_CAPTIONS: dict[str, str] | None = None


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


def load_english_captions() -> dict[str, str]:
    caption_paths = sorted(ENGLISH_CAPTIONS_DIR.glob("captions_task*.jsonl"))
    if not caption_paths:
        raise FileNotFoundError(f"No English caption files found under {ENGLISH_CAPTIONS_DIR}")

    captions: dict[str, str] = {}
    for path in caption_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                sample_id = payload["sample_id"]
                caption = normalize_caption(payload["caption"])
                if not sample_id or not caption:
                    raise ValueError(f"Invalid English caption record in {path}:{line_number}")
                if sample_id in captions:
                    raise ValueError(f"Duplicate English caption for {sample_id!r}")
                captions[sample_id] = caption

    return captions


def get_english_captions() -> dict[str, str]:
    global _ENGLISH_CAPTIONS
    if _ENGLISH_CAPTIONS is None:
        _ENGLISH_CAPTIONS = load_english_captions()
        print(f"Loaded {_ENGLISH_CAPTIONS.__len__()} English captions from {ENGLISH_CAPTIONS_DIR}")
    return _ENGLISH_CAPTIONS


def build_prompt(sample: dict) -> str:
    caption_en = normalize_caption(sample.get("conditioning_caption_en"))
    if not caption_en:
        raise ValueError(f"Sample {sample['sample_id']!r} is missing the English recap caption")

    return PROMPT_TEMPLATE.format(
        language_name=TARGET_CONFIG["name"],
        caption_en=caption_en,
    )


def iter_samples_from_shard(shard_path: Path) -> Iterator[dict]:
    english_captions = get_english_captions()
    dataset = wds.WebDataset(str(shard_path), shardshuffle=False, empty_check=False)

    for sample in dataset:
        image_bytes, media_type = extract_image(sample)
        raw_metadata = sample.get("json", b"{}")
        metadata = json.loads(bytes(raw_metadata).decode("utf-8"))
        sample_id = sample["__key__"]
        metadata_key = metadata.get("ImageID")
        if metadata_key is not None and metadata_key != sample_id:
            raise ValueError(f"Mismatched ImageID for {shard_path}: sample={sample_id!r} metadata={metadata_key!r}")

        try:
            conditioning_caption_en = english_captions[sample_id]
        except KeyError as exc:
            raise KeyError(f"Missing English recap caption for sample_id={sample_id!r}") from exc

        yield {
            "sample_id": sample_id,
            "image_bytes": image_bytes,
            "media_type": media_type,
            "conditioning_caption_en": conditioning_caption_en,
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
    name=f"mitsua_art_museums_pd_440k_qwen_from_en___{TARGET_LANG}",
    output_dir=OUTPUT_DIR,
    system_prompt=SYSTEM_PROMPT,
    build_prompt=build_prompt,
    iter_samples=iter_samples,
)
