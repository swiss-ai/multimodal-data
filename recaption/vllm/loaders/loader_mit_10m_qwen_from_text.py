import json
import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from vllm import SamplingParams

WORKDIR = Path(__file__).resolve().parent
DATASET_ROOT = Path("/path/to/data/vision-datasets/MIT-10M")
IMAGE_TIER = "big"
IMAGE_ARCHIVE_DIR = DATASET_ROOT / "data" / IMAGE_TIER
MANIFEST_ROOT = WORKDIR / "manifests" / "mit_10m_big"
MODEL_DIR = "/tmp/models"
MODEL_REPO = "Qwen/Qwen3.5-9B"
MODEL_CACHE_DIR = Path(MODEL_DIR) / "models--Qwen--Qwen3.5-9B"
LANGUAGE_CONFIGS = {
    "ar": {"name": "Arabic", "output_subdir": "ar"},
    "de": {"name": "German", "output_subdir": "de"},
    "en": {"name": "English", "output_subdir": "en"},
    "es": {"name": "Spanish", "output_subdir": "es"},
    "fr": {"name": "French", "output_subdir": "fr"},
    "hi": {"name": "Hindi", "output_subdir": "hi"},
    "it": {"name": "Italian", "output_subdir": "it"},
    "ja": {"name": "Japanese", "output_subdir": "ja"},
    "ko": {"name": "Korean", "output_subdir": "ko"},
    "pt": {"name": "Portuguese", "output_subdir": "pt"},
    "ru": {"name": "Russian", "output_subdir": "ru"},
    "th": {"name": "Thai", "output_subdir": "th"},
    "tr": {"name": "Turkish", "output_subdir": "tr"},
    "zh": {"name": "Chinese", "output_subdir": "zh"},
}
TARGET_LANG = os.environ["RECAPTION_TARGET_LANG"]
try:
    TARGET_CONFIG = LANGUAGE_CONFIGS[TARGET_LANG]
except KeyError as exc:
    raise KeyError(
        f"Unsupported RECAPTION_TARGET_LANG={TARGET_LANG!r}; expected one of {sorted(LANGUAGE_CONFIGS)}"
    ) from exc

OUTPUT_DIR = WORKDIR / "outputs" / "mit_10m" / TARGET_CONFIG["output_subdir"]
PROMPT_TEMPLATE = """Write one cohesive {language_name} paragraph.
Use the image as the primary source of truth.
Use the provided {language_name} text only as a supporting hint for the meaning of labels, packaging copy, interface text, signage, or printed claims when it matches the image.
Start directly with the main product or scene itself.
Do not use meta framing such as "in this image", "this picture shows", "this product showcase", "this layout", "the panel highlights", "overall", or similar openings or summaries.
Describe the object or scene itself in fluent prose, not the fact that it appears in an advertisement, comparison chart, feature card, or collage.
Mention callouts, panels, labels, or text placement only when they are clearly visible and genuinely helpful.
Describe concrete visual details such as shape, color, material, parts, packaging or interface design, icons, and any hands, people, or background elements that are clearly visible.
If the provided text is broken into bullet points or short lines, fold it into natural prose rather than copying the list structure or line breaks.
Do not add promotional language, quality judgments, or generic closing sentences.
Do not claim that the exact {language_name} words are visibly printed unless the image clearly supports that. Treat the text as semantic context, not guaranteed visual transcription.
Do not invent brands, features, or details unsupported by the image and the provided text.
Return only the final paragraph in {language_name}.

Provided {language_name} text:
{conditioning_text}
"""
PROMPT_TEMPLATE_NO_TEXT = """Write one cohesive {language_name} paragraph.
Use the image as the primary source of truth.
No supporting text is available for this sample, so rely only on the visible content of the image.
Start directly with the main product or scene itself.
Do not use meta framing such as "in this image", "this picture shows", "this product showcase", "this layout", "the panel highlights", "overall", or similar openings or summaries.
Describe the object or scene itself in fluent prose, not the fact that it appears in an advertisement, comparison chart, feature card, or collage.
Mention callouts, panels, labels, or text placement only when they are clearly visible and genuinely helpful.
Describe concrete visual details such as shape, color, material, parts, packaging or interface design, icons, and any hands, people, or background elements that are clearly visible.
Do not add promotional language, quality judgments, or generic closing sentences.
Do not invent brands, features, or details unsupported by the image.
Return only the final paragraph in {language_name}.
"""
_ZIP_HANDLES: dict[str, ZipFile] = {}


def normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return "\n".join(part.strip() for part in value.splitlines() if part.strip())


def get_manifest_path(task_id: int, task_count: int) -> Path:
    if task_count != 64:
        raise ValueError(f"MIT-10M manifests were built for 64 logical tasks; got {task_count}")
    path = MANIFEST_ROOT / TARGET_LANG / f"manifest_task{task_id:04d}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest shard: {path}")
    return path


def get_zip_handle(archive_lang: str) -> ZipFile:
    handle = _ZIP_HANDLES.get(archive_lang)
    if handle is None:
        archive_path = IMAGE_ARCHIVE_DIR / f"{archive_lang}.zip"
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing image archive: {archive_path}")
        handle = ZipFile(archive_path)
        _ZIP_HANDLES[archive_lang] = handle
    return handle


def build_prompt(sample: dict) -> str:
    conditioning_text = normalize_text(sample.get("conditioning_text"))
    if not conditioning_text:
        return PROMPT_TEMPLATE_NO_TEXT.format(
            language_name=TARGET_CONFIG["name"],
        )
    return PROMPT_TEMPLATE.format(
        language_name=TARGET_CONFIG["name"],
        conditioning_text=conditioning_text,
    )


def iter_samples(task_id: int, task_count: int) -> Iterator[dict]:
    manifest_path = get_manifest_path(task_id, task_count)
    print(f"Reading manifest shard {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            img_path = payload["img"]
            archive_lang = payload["image_archive_lang"]
            zip_handle = get_zip_handle(archive_lang)
            with zip_handle.open(img_path) as image_handle:
                image_bytes = image_handle.read()

            yield {
                "sample_id": payload["sample_id"],
                "image_bytes": image_bytes,
                "media_type": "image/jpeg",
                "conditioning_text": payload["conditioning_text"],
                "metadata": {
                    "img": img_path,
                    "image_archive_lang": archive_lang,
                    "conditioning_lang": payload["conditioning_lang"],
                    "text_role": payload["text_role"],
                    "box_cnt": payload["box_cnt"],
                    "difficulty": payload["difficulty"],
                    "cate_id": payload["cate_id"],
                    "cate_name": payload["cate_name"],
                    "split": payload["split"],
                    "row_id": payload["row_id"],
                    "manifest": str(manifest_path),
                    "manifest_line": line_number,
                },
            }


loader = SimpleNamespace(
    name=f"mit_10m_qwen_from_text___{TARGET_LANG}",
    output_dir=OUTPUT_DIR,
    build_prompt=build_prompt,
    iter_samples=iter_samples,
    model_dir=MODEL_DIR,
    model_repo=MODEL_REPO,
    model_cache_dir=MODEL_CACHE_DIR,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.94,
    max_model_len=16384,
    batch_size=64,
    prefetch_batches=1,
    max_num_batched_tokens=6144,
    mm_processor_kwargs={
        "min_pixels": 28 * 28,
        "max_pixels": 1280 * 28 * 28,
    },
    chat_template_kwargs={"enable_thinking": False},
    sampling_params=SamplingParams(
        temperature=0.25,
        top_p=0.8,
        top_k=20,
        max_tokens=220,
    ),
)
