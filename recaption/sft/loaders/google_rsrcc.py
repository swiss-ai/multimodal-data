from __future__ import annotations

import csv
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sft_recaption.loaders.base import BaseLoader
from sft_recaption.prompts import (
    RSRCC_COT_PROMPT_VERSION,
    RSRCC_COT_SCHEMA,
    build_rsrcc_cot_instruction,
)
from sft_recaption.schemas import (
    CandidateExample,
    ImagePayload,
    ManifestRecord,
    SourceSample,
)

DATASET_ROOT = "/path/to/data/vision-datasets/raw/sft/hf___google___RSRCC"

# "train+val" combines both splits; individual splits are "train" and "val".
KNOWN_SPLITS = {"train", "val", "train+val"}


def _parse_question_answer(text: str) -> tuple[str, str]:
    """Extract question and answer from the metadata text field.

    Handles both simple Q&A and multiple-choice formats:
      **Question:** ...\n\n**Answer:** Yes
      **Question:** ...\n\n**A)** ...\n**B)** ...\n\n**Answer:** A

    The CSV stores newlines as literal backslash-n sequences; we decode them here.
    """
    # Decode literal \n escape sequences stored in the CSV
    text = text.replace("\\n", "\n")
    # Split on **Answer:** to separate question block from answer
    parts = re.split(r"\*\*Answer:\*\*\s*", text, maxsplit=1)
    if len(parts) == 2:
        question_block = re.sub(r"^\*\*Question:\*\*\s*", "", parts[0]).strip()
        answer = parts[1].strip()
        return question_block, answer
    # Fallback: treat whole text as question, no answer
    return text.strip(), ""


class GoogleRsrccLoader(BaseLoader):
    """Loader for the Google RSRCC (Remote Sensing Change Detection) dataset.

    Dataset structure (per split directory):
      images/{shard}/{uuid}_before.png
      images/{shard}/{uuid}_after.png
      metadata.csv  — columns: before_file_name, after_file_name, text

    The "text" column contains a question+answer pair.  We generate CoT
    reasoning that organically bridges the two satellite images to that answer.

    Supported --split values: "train", "val", "train+val".
    Production run uses task_count = 32 nodes * 4 GPUs = 128.
    """

    name = "google_rsrcc"
    dataset_path = DATASET_ROOT
    default_split = "train"

    def dataset_root(self) -> Path:
        return Path(os.environ.get("SFT_RECAPTION_RSRCC_DATASET_PATH", self.dataset_path))

    def split_dir(self, split: str) -> Path:
        return self.dataset_root() / split

    def _read_csv_rows(self, split: str) -> list[dict[str, str]]:
        csv_path = self.split_dir(split) / "metadata.csv"
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [dict(row) for row in reader]

    def _splits_to_process(self, target_split: str) -> list[str]:
        if target_split == "train+val":
            return ["train", "val"]
        if target_split in KNOWN_SPLITS:
            return [target_split]
        raise ValueError(f"Unsupported split {target_split!r} for {self.name!r}; use one of {sorted(KNOWN_SPLITS)}")

    # ------------------------------------------------------------------ #
    # Manifest writing (replaces HF dataset approach)
    # ------------------------------------------------------------------ #

    def load_dataset(self):  # type: ignore[override]
        raise NotImplementedError(f"{self.name} reads CSV files directly; use prepare-manifests / generate")

    def write_manifests(
        self,
        output_dir: Path,
        task_count: int,
        split: str | None = None,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        target_split = split or self.default_split
        handles: list[tuple[Path, Any]] = []
        try:
            for task_id in range(task_count):
                path = output_dir / f"manifest_task{task_id:04d}.jsonl"
                handles.append((path, path.open("w", encoding="utf-8")))

            row_index = 0
            for current_split in self._splits_to_process(target_split):
                split_dir = self.split_dir(current_split)
                for row in self._read_csv_rows(current_split):
                    before_rel = row["before_file_name"]
                    after_rel = row["after_file_name"]
                    before_abs = str(split_dir / before_rel)
                    after_abs = str(split_dir / after_rel)
                    question, answer = _parse_question_answer(row["text"])

                    # Stable ID from the before-image stem (contains UUID)
                    before_stem = Path(before_rel).stem  # e.g. 8d867bbe_24b2_..._before
                    sample_id = f"{current_split}:{before_stem}"

                    record = ManifestRecord(
                        sample_id=sample_id,
                        split=target_split,
                        row_index=row_index,
                        metadata={
                            "before_path": before_abs,
                            "after_path": after_abs,
                            "question": question,
                            "answer": answer,
                            "source_split": current_split,
                        },
                    )
                    task_id_int = self.stable_task_id(sample_id, task_count)
                    handles[task_id_int][1].write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                    row_index += 1
        finally:
            for _, fh in handles:
                fh.close()
        return [path for path, _ in handles]

    # ------------------------------------------------------------------ #
    # Sample loading
    # ------------------------------------------------------------------ #

    def get_source_sample(self, record: ManifestRecord) -> SourceSample:
        if record.metadata is None:
            raise ValueError(f"{self.name} manifest record {record.sample_id!r} has no metadata")
        before_image = self.encode_image(record.metadata["before_path"])
        after_image = self.encode_image(record.metadata["after_path"])
        source_split = record.metadata.get("source_split", record.split)
        question = record.metadata.get("question", "")
        answer = record.metadata.get("answer", "")
        metadata = {
            "source_dataset": self.name,
            "source_split": source_split,
            "source_doc_id": record.sample_id,
            "data_source": "google/RSRCC",
            "license": None,
            "url": None,
            "source_fields_json": json.dumps(
                {
                    "before_path": record.metadata.get("before_path", ""),
                    "after_path": record.metadata.get("after_path", ""),
                    "question": question,
                    "answer": answer,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            # Carried for prompt construction; stripped before writing to candidates
            "_question": question,
            "_answer": answer,
        }
        return SourceSample(
            sample_id=record.sample_id,
            split=record.split,
            row_index=record.row_index,
            images=[before_image, after_image],
            metadata=metadata,
        )

    # BaseLoader abstract methods (unused — manifest/source_sample are overridden above)
    def build_sample_id(self, split: str, row_index: int, row: dict[str, Any]) -> str:
        return f"{split}:{row_index}"

    def extract_images(self, row: dict[str, Any]) -> list[ImagePayload]:
        raise NotImplementedError(f"{self.name} overrides get_source_sample directly")

    # ------------------------------------------------------------------ #
    # Generation hooks
    # ------------------------------------------------------------------ #

    @property
    def generation_schema(self) -> dict:
        return RSRCC_COT_SCHEMA

    @property
    def generation_prompt_version(self) -> str:
        return RSRCC_COT_PROMPT_VERSION

    def build_generation_instruction_text(self, sample: SourceSample) -> str:
        question = sample.metadata.get("_question", "")
        answer = sample.metadata.get("_answer", "")
        return build_rsrcc_cot_instruction(question=question, answer=answer)

    def _normalize_payload(self, payload: Any) -> Any:
        if isinstance(payload, dict) and isinstance(payload.get("reasoning"), str):
            return {**payload, "reasoning": payload["reasoning"].strip()}
        return payload

    def _build_candidates(
        self,
        sample: SourceSample,
        payload: Any,
        *,
        generator_model: str,
    ) -> list[CandidateExample]:
        question = sample.metadata.get("_question", "")
        reasoning = payload["reasoning"].strip()
        image_prefix = "\n".join(f"<image_{i}>" for i in range(len(sample.images)))
        created_at = datetime.now(UTC).isoformat()

        # Metadata written to JSONL — strip internal _ keys used only for prompting.
        # source_fields_json already contains question + answer + image paths,
        # so the WebDataset writer can reconstruct everything without parsing messages.
        shared_metadata = {k: v for k, v in sample.metadata.items() if not k.startswith("_")}
        shared_metadata.update(
            {
                "generator_model": generator_model,
                "prompt_version": RSRCC_COT_PROMPT_VERSION,
                "judge_model": None,
                "judge_prompt_version": None,
                "created_at_utc": created_at,
            }
        )
        return [
            CandidateExample(
                sample_id=f"{sample.sample_id}::cot_0",
                source_sample_id=sample.sample_id,
                source_split=sample.split,
                source_row_index=sample.row_index,
                task_type="cot_vqa",
                # messages[0]: user sees images + question
                # messages[1]: reasoning ONLY (no <think> tags, no final answer)
                #   → tags + source answer are appended at WebDataset export time
                messages=[
                    {
                        "role": "user",
                        "content": f"{image_prefix}\n{question}",
                    },
                    {
                        "role": "assistant",
                        "content": reasoning,
                    },
                ],
                metadata=shared_metadata,
            )
        ]
