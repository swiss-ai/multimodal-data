import json
import tarfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

from vllm import SamplingParams

SYSTEM_PROMPT = "You are an expert image analyst and visual writer. Return only the description text in plain prose."
DESCRIPTIVE_PROMPT_TEMPLATE = """You are an expert image analyst. Your task is to provide a highly detailed, comprehensive, and accurate description of the provided image.
Use the image as the ground truth. The short reference caption and subject anchor below are only supporting hints; ignore any hint that is not visibly supported by the image.

Reference hints:
{reference_hints}

Follow these strict guidelines:
- Directness: Do not use introductory filler phrases like "This image shows", "A picture of", or "Here we can see". Start describing immediately.
- Main Subject: Describe the main subject or subjects in extreme detail, including appearance, materials, clothing, colors, textures, markings, and actions when relevant.
- Composition: Explicitly state the spatial relationships between objects, such as "behind", "on top of", "to the left", "in the foreground", and "in the distance".
- Environment: Describe the background, setting, and any secondary objects.
- Atmosphere & Style: Mention the lighting, the mood, and the medium or style, such as photograph, digital art, oil painting, or 3D render.
- Text: If there is any readable text in the image, quote it exactly.
- Specificity: Focus on the details that make this exact view distinct, including angle, framing, pose, setting, weather, decor, and background differences rather than writing a generic description of the subject.
Output one or more cohesive paragraphs that seamlessly integrate all these details.
Make the description really long while staying faithful to what is visibly present.
Do not guess beyond the image.
Do not mention the prompt, the user, your reasoning, or meta phrases like "this image shows"."""

CREATIVE_PROMPT_TEMPLATE = """You are writing a long, vivid piece of observational prose about the provided image.
Use the image as the ground truth. The short reference caption and subject anchor below are only supporting hints; ignore any hint that is not visibly supported by the image.

Reference hints:
{reference_hints}

Follow these strict guidelines:
- Write in a more expressive, flowing style than a standard catalog description, but stay fully grounded in visible evidence.
- Do not invent backstory, symbolism, or events that are not visible.
- Do not use introductory filler phrases like "This image shows", "A picture of", or "Here we can see".
- Favor rhythm, atmosphere, texture, and scene-setting over list-like enumeration.
- Describe the main subject in detail, but let the surrounding setting, lighting, color, and mood shape the paragraph.
- Mention spatial relationships when they matter to the scene, but do so naturally inside the prose rather than as a checklist.
- If any readable text appears, quote it exactly.
- Keep the output long. One or more substantial paragraphs are welcome.
- Make this view feel distinct from another possible view of the same subject by leaning into the specific framing, environment, and mood that are actually visible.
Do not mention the prompt, the user, your reasoning, or meta phrases like "this image shows"."""

DATASET_ROOT = Path("/path/to/data/vision-datasets/UNO-1M")
LABEL_DIR = DATASET_ROOT / "labels"
IMAGE_DIR = DATASET_ROOT / "images"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "uno_1m_v3"
IMAGE_SUFFIX_TO_MEDIA_TYPE = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def normalize_subjects(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    subjects = []
    seen = set()
    for item in value:
        text = normalize_text(item)
        key = text.casefold()
        if text and key not in seen:
            subjects.append(text)
            seen.add(key)
    return subjects


def iter_label_paths() -> list[Path]:
    paths = sorted(LABEL_DIR.glob("split*.json"))
    if not paths:
        raise FileNotFoundError(f"No UNO label shards found under {LABEL_DIR}")
    return paths


def media_type_for_path(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    media_type = IMAGE_SUFFIX_TO_MEDIA_TYPE.get(suffix)
    if media_type is None:
        raise ValueError(f"Unsupported image suffix for {image_path!r}")
    return media_type


def build_reference_hints(sample: dict) -> str:
    reference_lines = []
    original_caption = normalize_text(sample["metadata"].get("original_caption"))
    if original_caption:
        reference_lines.append(f"- Short reference caption: {original_caption}")

    subjects = sample["metadata"].get("subject") or []
    if subjects:
        reference_lines.append(f"- Subject anchor: {', '.join(subjects)}")

    if not reference_lines:
        reference_lines.append("- No reliable text hint is available.")
    return "\n".join(reference_lines)


def build_prompt(sample: dict) -> str:
    reference_hints = build_reference_hints(sample)
    image_role = sample["metadata"].get("image_role")
    template = DESCRIPTIVE_PROMPT_TEMPLATE if image_role == "img_path1" else CREATIVE_PROMPT_TEMPLATE
    return template.format(reference_hints=reference_hints)


def iter_samples_from_split(label_path: Path) -> Iterator[dict]:
    split_name = label_path.stem
    archive_path = IMAGE_DIR / f"{split_name}.tar.gz"
    rows = json.loads(label_path.read_text(encoding="utf-8"))
    expected = {}

    for row_index, row in enumerate(rows):
        caption = row.get("caption") or {}
        subjects = normalize_subjects(caption.get("subject"))
        score_bundle = row.get("vlm_filter_cot") or {}
        common_metadata = {
            "split": split_name,
            "archive_path": str(archive_path),
            "row_index": row_index,
            "pair_id": f"{row['img_path1']}||{row['img_path2']}",
            "subject": subjects,
            "judgment": normalize_text(caption.get("judgment")),
            "score_final": float(score_bundle.get("score_final", 0.0)),
            "score_part": score_bundle.get("score_part") or {},
        }
        expected[row["img_path1"]] = {
            "sample_id": row["img_path1"],
            "paired_image_path": row["img_path2"],
            "image_role": "img_path1",
            "original_caption": normalize_text(caption.get("img_path1")),
            "metadata": common_metadata,
        }
        expected[row["img_path2"]] = {
            "sample_id": row["img_path2"],
            "paired_image_path": row["img_path1"],
            "image_role": "img_path2",
            "original_caption": normalize_text(caption.get("img_path2")),
            "metadata": common_metadata,
        }

    with tarfile.open(archive_path, "r:gz") as handle:
        for member in handle:
            if not member.isfile():
                continue
            sample = expected.pop(member.name, None)
            if sample is None:
                continue

            extracted = handle.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Failed to read {member.name} from {archive_path}")

            metadata = dict(sample["metadata"])
            metadata.update(
                {
                    "source_tar": str(archive_path),
                    "image_path": sample["sample_id"],
                    "paired_image_path": sample["paired_image_path"],
                    "image_role": sample["image_role"],
                    "original_caption": sample["original_caption"],
                }
            )

            yield {
                "sample_id": sample["sample_id"],
                "image_bytes": extracted.read(),
                "media_type": media_type_for_path(sample["sample_id"]),
                "metadata": metadata,
            }

    if expected:
        missing = ", ".join(sorted(expected)[:5])
        raise FileNotFoundError(
            f"Missing {len(expected)} labeled UNO images in {archive_path}; first missing: {missing}"
        )


def iter_samples(task_id: int, task_count: int) -> Iterator[dict]:
    label_paths = iter_label_paths()
    total = len(label_paths)
    assigned_paths = label_paths[total * task_id // task_count : total * (task_id + 1) // task_count]
    print(
        f"Discovered {len(label_paths)} UNO label splits under {LABEL_DIR}; "
        f"task {task_id}/{task_count - 1} assigned {len(assigned_paths)}"
    )
    if assigned_paths:
        print(f"Assigned first split: {assigned_paths[0]}")
        print(f"Assigned last split:  {assigned_paths[-1]}")

    for label_path in assigned_paths:
        yield from iter_samples_from_split(label_path)


loader = SimpleNamespace(
    name="uno_1m_v3",
    output_dir=OUTPUT_DIR,
    build_prompt=build_prompt,
    system_prompt=SYSTEM_PROMPT,
    iter_samples=iter_samples,
    batch_size=48,
    sampling_params=SamplingParams(
        temperature=0.75,
        top_p=0.95,
        top_k=50,
        max_tokens=896,
    ),
)
