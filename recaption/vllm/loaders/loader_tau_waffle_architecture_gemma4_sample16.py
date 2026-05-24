from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import webdataset as wds
from vllm import SamplingParams

WORKDIR = Path(__file__).resolve().parent
_LOCAL_ROOT = WORKDIR / "outputs" / "tau_waffle_architecture_gemma4_sample16"
DATASET_ROOT = _LOCAL_ROOT / "sample_webdataset"
OUTPUT_ROOT = Path("/tmp/test")
OUTPUT_DIR = OUTPUT_ROOT / "captions"
MODEL_DIR = "/tmp/models"
MODEL_REPO = "google/gemma-4-31B-it"
MODEL_CACHE_DIR = Path(MODEL_DIR) / "models--google--gemma-4-31B-it"
MODEL_USED = str(MODEL_CACHE_DIR)
THINKING_ENABLED = False
IMAGE_KEYS_TO_MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}


def persona_for(high_level_type: str) -> str:
    value = (high_level_type or "").lower()
    if "historic" in value:
        return "an architectural historian specializing in preservation documentation"
    if "religious" in value:
        return "a scholar of sacred architecture"
    if "public" in value or "commercial" in value:
        return "an urban architecture writer"
    if "infrastructure" in value or "transportation" in value or "healthcare" in value:
        return "a civil engineering journalist"
    if "industrial" in value:
        return "an industrial heritage researcher"
    if "palaces" in value or "mansions" in value or "residential" in value:
        return "a decorative-arts historian focused on domestic architecture"
    if "institutional" in value:
        return "an architectural historian focused on institutional buildings"
    if "educational" in value:
        return "an architectural historian focused on educational buildings"
    if "castles" in value or "fortresses" in value:
        return "a military-architecture historian"
    return "an architectural writer"


def build_system_prompt(persona: str) -> str:
    return f"""You are {persona} writing a caption for a vision-language training dataset of architectural images.

Return exactly two sections:

### Perspective
Write 2-3 first-person sentences in the voice of {persona} describing your analytic stance for this image type. This is not the image description yet.

### Caption
Write 6-10 sentences enumerating visible architectural features, materials, legible labels or dimensions, panel composition, stylistic cues, and drawing or photographic technique.

Rules:
- Use the image as primary evidence and metadata only as supporting context.
- Only mention names, locations, or identifiers when they are visibly present in the image.
- Do not copy metadata phrasing unless it is legible in the image.
- Avoid flowery language and stay concrete.
"""


def build_user_prompt(metadata: dict) -> str:
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
    return f"""Write the response in the required two-section format.

Use the metadata below as supplemental context only. Include full metadata awareness in your reasoning, but keep the actual written output grounded in the visible image. When metadata conflicts with the image or cannot be visually verified, prefer the image.

Full metadata:
```json
{metadata_json}
```"""


def prepare_prompt_bundle(base_metadata: dict) -> dict[str, str]:
    persona = persona_for(base_metadata.get("high_level_building_type") or "")
    return {
        "prompt_persona": persona,
        "prompt_system": build_system_prompt(persona),
        "prompt_user": build_user_prompt(base_metadata),
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


def build_messages(raw_sample: dict) -> list[dict]:
    metadata = dict(raw_sample["metadata"])
    system_prompt = metadata["prompt_system"]
    user_prompt = metadata["prompt_user"]
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": raw_sample["image_data_url"],
                    },
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def to_data_url(image_bytes: bytes, media_type: str) -> str:
    import base64

    return f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def iter_samples_from_shard(shard_path: Path) -> Iterator[dict]:
    dataset = wds.WebDataset(str(shard_path), shardshuffle=False, empty_check=False)

    for sample in dataset:
        image_bytes, media_type = extract_image(sample)
        base_metadata = json.loads(bytes(sample["json"]).decode("utf-8"))
        prompt_bundle = prepare_prompt_bundle(base_metadata)
        metadata = {
            **base_metadata,
            **prompt_bundle,
            "source_tar": sample.get("__url__", str(shard_path)),
            "loader_name": "tau_waffle_architecture_gemma4_sample16",
            "model_used": MODEL_USED,
            "model_repo": MODEL_REPO,
            "thinking_enabled": THINKING_ENABLED,
            "prompt_style": "persona_perspective_caption_v1",
        }
        yield {
            "sample_id": metadata["sample_id"],
            "image_bytes": image_bytes,
            "media_type": media_type,
            "image_data_url": to_data_url(image_bytes, media_type),
            "metadata": metadata,
        }


def iter_samples(task_id: int, task_count: int) -> Iterator[dict]:
    shard_paths = iter_shard_paths()
    print(
        f"Discovered {len(shard_paths)} shards under {DATASET_ROOT}; "
        f"task {task_id}/{task_count - 1} using sample-index partitioning"
    )
    if shard_paths:
        print(f"First shard: {shard_paths[0]}")
        print(f"Last shard:  {shard_paths[-1]}")

    sample_index = 0
    for shard_path in shard_paths:
        for raw_sample in iter_samples_from_shard(shard_path):
            if sample_index % task_count != task_id:
                sample_index += 1
                continue
            sample_index += 1
            yield raw_sample


loader = SimpleNamespace(
    name="tau_waffle_architecture_gemma4_sample16",
    output_dir=OUTPUT_DIR,
    build_messages=build_messages,
    iter_samples=iter_samples,
    inference_backend="transformers",
    model_dir=MODEL_DIR,
    model_repo=MODEL_REPO,
    model_cache_dir=MODEL_CACHE_DIR,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.88,
    max_model_len=32768,
    batch_size=1,
    prefetch_batches=1,
    max_num_batched_tokens=16384,
    mm_processor_kwargs={
        "min_pixels": 28 * 28,
        "max_pixels": 1280 * 28 * 28,
    },
    chat_template_kwargs={"enable_thinking": THINKING_ENABLED},
    transformers_model_kwargs={},
    sampling_params=SamplingParams(
        temperature=0.0,
        top_p=0.85,
        top_k=30,
        max_tokens=512,
    ),
)
