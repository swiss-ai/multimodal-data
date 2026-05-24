from __future__ import annotations

import json
import os
import random
import tarfile
from datetime import UTC, datetime
from functools import cached_property
from hashlib import blake2b
from pathlib import Path
from typing import Any

from sft_recaption.loaders.base import BaseLoader
from sft_recaption.schemas import (
    CandidateExample,
    ImagePayload,
    ManifestRecord,
    SourceSample,
)

HQ50K_ROOT = Path("/path/to/data/vision-datasets/raw/cooldown/hf___YangQiee___HQ-50K/downloaded")
DOCCI_ARROW = Path("/path/to/data/vision-datasets/raw/stage2/hf___google___docci/docci-train.arrow")
DOCCI_IMAGES_DIR = DOCCI_ARROW.parent / "images"

N_DOCCI_EXAMPLES = 10

CAPTION_PROMPT = (
    "You are describing an image for a dataset. "
    "Begin with a noun phrase that identifies the subject. "
    "Then give a factual, detailed account of everything visible — "
    "objects, people, text, setting, and their relationships. "
    "Adjust the length to the complexity of the image."
)

GENERATION_PROMPT_VERSION = "hq50k_caption_docci10_v1"


class HQ50KLoader(BaseLoader):
    name = "hq50k"
    dataset_path = str(HQ50K_ROOT)
    default_split = "train"

    def load_dataset(self):
        raise NotImplementedError("hq50k uses tar shards directly; use prepare-manifests/generate")

    def _shard_paths(self) -> list[Path]:
        root = Path(os.environ.get("SFT_RECAPTION_HQ50K_ROOT", str(HQ50K_ROOT)))
        return sorted(root.glob("*.tar"))

    @cached_property
    def docci_index(self) -> list[dict]:
        """Load DOCCI caption index once per worker process."""
        import pyarrow.ipc as ipc

        with ipc.open_stream(DOCCI_ARROW) as f:
            table = f.read_all()
        index = []
        for i in range(len(table)):
            example_id = table["example_id"][i].as_py()
            img_path = DOCCI_IMAGES_DIR / f"{example_id}.jpg"
            if img_path.exists():
                index.append(
                    {
                        "img_path": img_path,
                        "caption": table["description"][i].as_py(),
                    }
                )
        return index

    # --- Manifest writing ---

    def write_manifests(
        self,
        output_dir: Path,
        task_count: int,
        split: str | None = None,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        target_split = split or self.default_split
        handles: list[tuple[Path, Any]] = []
        row_index = 0
        try:
            for task_id in range(task_count):
                path = output_dir / f"manifest_task{task_id:04d}.jsonl"
                handles.append((path, path.open("w", encoding="utf-8")))
            for shard_path in self._shard_paths():
                with tarfile.open(shard_path, mode="r") as tf:
                    jpg_members = [m for m in tf.getmembers() if m.name.endswith(".jpg")]
                for member in jpg_members:
                    sample_id = member.name.replace(".jpg", "")
                    record = ManifestRecord(
                        sample_id=sample_id,
                        split=target_split,
                        row_index=row_index,
                        metadata={
                            "tar_path": str(shard_path),
                            "member_name": member.name,
                            "shard_name": shard_path.name,
                        },
                    )
                    tid = self.stable_task_id(sample_id, task_count)
                    handles[tid][1].write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                    row_index += 1
        finally:
            for _, handle in handles:
                handle.close()
        return [path for path, _ in handles]

    # --- Sample loading ---

    def get_source_sample(self, record: ManifestRecord) -> SourceSample:
        if record.metadata is None:
            raise ValueError("HQ50K manifest record is missing tar shard metadata")
        tar_path = record.metadata["tar_path"]
        member_name = record.metadata["member_name"]
        shard_name = record.metadata["shard_name"]
        with tarfile.open(tar_path, mode="r") as tf:
            file_obj = tf.extractfile(member_name)
            if file_obj is None:
                raise FileNotFoundError(f"Could not extract {member_name} from {tar_path}")
            image_bytes = file_obj.read()
        image = ImagePayload(media_type="image/jpeg", data=image_bytes)
        metadata = {
            "source_dataset": self.name,
            "source_split": record.split,
            "source_doc_id": record.sample_id,
            "data_source": "YangQiee/HQ-50K",
            "license": None,
            "url": None,
            "source_fields_json": json.dumps(
                {"shard_name": shard_name, "member_name": member_name},
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        return SourceSample(
            sample_id=record.sample_id,
            split=record.split,
            row_index=record.row_index,
            images=[image],
            metadata=metadata,
        )

    # --- Generation ---

    @property
    def generation_prompt_version(self) -> str:
        return GENERATION_PROMPT_VERSION

    @property
    def generation_sampling_params(self) -> dict:
        return {"temperature": 0.7, "top_p": 0.9, "max_tokens": 1024}

    @property
    def limit_mm_per_prompt(self) -> dict[str, int] | None:
        return {"image": N_DOCCI_EXAMPLES + 1}

    def build_generation_messages(self, sample: SourceSample) -> list[dict]:
        from sft_recaption.runtime import to_data_url

        # Deterministic DOCCI selection per image (reproducible across retries)
        seed = int.from_bytes(blake2b(sample.sample_id.encode(), digest_size=8).digest(), "big")
        rng = random.Random(seed)
        examples = rng.sample(self.docci_index, N_DOCCI_EXAMPLES)

        messages: list[dict] = []
        for ex in examples:
            ex_payload = ImagePayload(
                media_type="image/jpeg",
                data=ex["img_path"].read_bytes(),
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": to_data_url(ex_payload)},
                        },
                        {"type": "text", "text": CAPTION_PROMPT},
                    ],
                }
            )
            messages.append({"role": "assistant", "content": ex["caption"]})

        target_payload = sample.images[0]
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": to_data_url(target_payload)},
                    },
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }
        )
        return messages

    def process_generation_output(
        self,
        sample: SourceSample,
        output_text: str,
        *,
        generator_model: str,
    ) -> list[CandidateExample]:
        caption = output_text.strip()
        if not caption:
            raise ValueError("Empty caption output")
        created_at = datetime.now(UTC).isoformat()
        metadata = {
            **sample.metadata,
            "generator_model": generator_model,
            "prompt_version": self.generation_prompt_version,
            "judge_model": None,
            "judge_prompt_version": None,
            "created_at_utc": created_at,
        }
        return [
            CandidateExample(
                sample_id=sample.sample_id,
                source_sample_id=sample.sample_id,
                source_split=sample.split,
                source_row_index=sample.row_index,
                task_type="caption",
                messages=[
                    {"role": "user", "content": "<image_0>"},
                    {"role": "assistant", "content": caption},
                ],
                metadata=metadata,
            )
        ]

    # --- Abstract method stubs (tar-based; not used via HF dataset path) ---

    def build_sample_id(self, split: str, row_index: int, row: dict[str, Any]) -> str:
        raise NotImplementedError("hq50k uses tar shards directly")

    def extract_images(self, row: dict[str, Any]) -> list[ImagePayload]:
        raise NotImplementedError("hq50k uses tar shards directly")
