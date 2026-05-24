from pathlib import Path
from types import SimpleNamespace

from loader_openimages_v7_dense___full import (
    DATASET_ROOT,
    iter_samples_from_shard,
)
from loader_openimages_v7_dense___full import (
    iter_shard_paths as base_iter_shard_paths,
)
from vllm import SamplingParams

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "openimages_v7_dense___full_v2"
SYSTEM_PROMPT = "You are an expert image analyst. Return only the description text in plain prose."
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


def iter_samples(task_id: int, task_count: int):
    shard_paths = base_iter_shard_paths()
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
    name="openimages_v7_dense___full_v2",
    output_dir=OUTPUT_DIR,
    prompt=PROMPT,
    system_prompt=SYSTEM_PROMPT,
    iter_samples=iter_samples,
    batch_size=64,
    sampling_params=SamplingParams(
        temperature=0.3,
        top_p=0.7,
        top_k=20,
        repetition_penalty=1.2,
        max_tokens=768,
    ),
)
