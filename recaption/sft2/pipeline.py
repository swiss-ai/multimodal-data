from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import pyarrow as pa
import pyarrow.parquet as pq
from sft_recaption.config import (
    CANDIDATES_DIR,
    CURATED_DIR,
    DEFAULT_CANDIDATES_PER_SAMPLE,
    GENERATION_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
    MANIFESTS_DIR,
    PARQUET_EXPORT_DIR,
)
from sft_recaption.json_utils import extract_json_object
from sft_recaption.loaders.base import BaseLoader
from sft_recaption.prompts import (
    GENERATION_SCHEMA,
    JUDGE_SCHEMA,
    build_generation_instruction,
    build_judge_instruction,
)
from sft_recaption.runtime import ChatEngine, to_data_url
from sft_recaption.schemas import (
    CandidateExample,
    CuratedExample,
    ManifestRecord,
    SourceSample,
)

BANNED_META_PHRASES = (
    "the user asks",
    "the prompt asks",
    "based on the image",
    "i can see",
    "as an ai",
)


def loader_manifest_dir(loader: BaseLoader) -> Path:
    return MANIFESTS_DIR / loader.name


def loader_candidates_dir(loader: BaseLoader) -> Path:
    return CANDIDATES_DIR / loader.name


def loader_curated_dir(loader: BaseLoader) -> Path:
    return CURATED_DIR / loader.name


def loader_export_dir(loader: BaseLoader) -> Path:
    return PARQUET_EXPORT_DIR / loader.name


def load_manifest_index(
    loader: BaseLoader,
) -> dict[tuple[str, str, int], ManifestRecord]:
    index: dict[tuple[str, str, int], ManifestRecord] = {}
    for path in sorted(loader_manifest_dir(loader).glob("manifest_task*.jsonl")):
        for record in loader.load_manifest(path):
            index[(record.sample_id, record.split, record.row_index)] = record
    return index


def prepare_manifests(
    loader: BaseLoader,
    *,
    task_count: int,
    split: str | None = None,
) -> list[Path]:
    manifest_dir = loader_manifest_dir(loader)
    existing_paths = sorted(manifest_dir.glob("manifest_task*.jsonl"))
    if len(existing_paths) == task_count and all(path.stat().st_size > 0 for path in existing_paths):
        return existing_paths
    for path in existing_paths:
        path.unlink()
    return loader.write_manifests(manifest_dir, task_count, split)


def make_generation_messages(sample: SourceSample) -> list[dict[str, object]]:
    content: list[dict[str, object]] = []
    for image in sample.images:
        content.append({"type": "image_url", "image_url": {"url": to_data_url(image)}})
    content.append({"type": "text", "text": build_generation_instruction()})
    return [{"role": "user", "content": content}]


def build_examples_from_bundle(
    sample: SourceSample,
    bundle: dict[str, Any],
    *,
    generator_model: str,
) -> list[CandidateExample]:
    examples: list[CandidateExample] = []
    image_prefix = "\n".join(f"<image_{index}>" for index in range(len(sample.images)))
    created_at = datetime.now(UTC).isoformat()
    shared_metadata = {
        **sample.metadata,
        "generator_model": generator_model,
        "prompt_version": GENERATION_PROMPT_VERSION,
        "judge_model": None,
        "judge_prompt_version": None,
        "created_at_utc": created_at,
    }
    for index, pair in enumerate(bundle["reasoning_qa"]):
        assistant_content = pair["response"].strip()
        examples.append(
            CandidateExample(
                sample_id=f"{sample.sample_id}::reasoning_qa_{index}",
                source_sample_id=sample.sample_id,
                source_split=sample.split,
                source_row_index=sample.row_index,
                task_type="reasoning_qa",
                messages=[
                    {
                        "role": "user",
                        "content": f"{image_prefix}\n{pair['question'].strip()}",
                    },
                    {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                ],
                metadata=dict(shared_metadata),
            )
        )
    return examples


def normalize_generation_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    pairs = payload.get("reasoning_qa")
    if not isinstance(pairs, list):
        return payload

    normalized_pairs: list[Any] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            normalized_pairs.append(pair)
            continue
        normalized_pair = dict(pair)
        final_answer = normalized_pair.pop("final_answer", None)
        response = normalized_pair.get("response")
        if isinstance(response, str) and isinstance(final_answer, str):
            response_text = response.strip()
            final_answer_text = final_answer.strip()
            if final_answer_text and final_answer_text not in response_text:
                joiner = "\n" if response_text else ""
                normalized_pair["response"] = f"{response_text}{joiner}{final_answer_text}"
            else:
                normalized_pair["response"] = response_text
        normalized_pairs.append(normalized_pair)

    normalized_payload = dict(payload)
    normalized_payload["reasoning_qa"] = normalized_pairs
    return normalized_payload


def generate_candidates(
    loader: BaseLoader,
    engine: ChatEngine,
    *,
    task_id: int,
    task_count: int,
    batch_size: int,
    split: str | None = None,
    limit: int | None = None,
    temperature: float = 0.2,
    top_p: float = 0.85,
    max_tokens: int = 700,
    model_repo: str,
) -> Path:
    output_path, pending_records = get_pending_generation_records(
        loader,
        task_id=task_id,
        task_count=task_count,
        split=split,
        limit=limit,
    )
    if not pending_records:
        return output_path

    failure_path = output_path.with_name(output_path.stem.replace("candidates_", "failures_") + ".jsonl")
    with output_path.open("a", encoding="utf-8", buffering=1) as handle:
        for batch_start in range(0, len(pending_records), batch_size):
            batch_records = pending_records[batch_start : batch_start + batch_size]
            samples = [loader.get_source_sample(record) for record in batch_records]
            conversations = [make_generation_messages(sample) for sample in samples]
            outputs = engine.chat(
                conversations,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            for sample, output_text in zip(samples, outputs):
                try:
                    payload = extract_json_object(output_text)
                    payload = normalize_generation_payload(payload)
                    jsonschema.validate(payload, GENERATION_SCHEMA)
                    examples = build_examples_from_bundle(
                        sample,
                        payload,
                        generator_model=model_repo,
                    )
                except Exception as exc:
                    failure_path.parent.mkdir(parents=True, exist_ok=True)
                    with failure_path.open("a", encoding="utf-8") as failure_handle:
                        failure_handle.write(
                            json.dumps(
                                {
                                    "sample_id": sample.sample_id,
                                    "source_split": sample.split,
                                    "source_row_index": sample.row_index,
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "raw_output": output_text,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    continue
                for example in examples:
                    handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
                handle.flush()
    return output_path


def get_pending_generation_records(
    loader: BaseLoader,
    *,
    task_id: int,
    task_count: int,
    split: str | None = None,
    limit: int | None = None,
) -> tuple[Path, list[ManifestRecord]]:
    manifest_path = loader_manifest_dir(loader) / f"manifest_task{task_id:04d}.jsonl"
    if not manifest_path.exists():
        loader.write_manifests(loader_manifest_dir(loader), task_count, split)
    records = loader.load_manifest(manifest_path)
    if limit is not None:
        records = records[:limit]
    output_dir = loader_candidates_dir(loader)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"candidates_task{task_id:04d}.jsonl"
    valid_source_ids = {record.sample_id for record in records}
    completed_source_ids = sanitize_existing_candidates(
        output_path,
        valid_source_ids=valid_source_ids,
    )
    pending_records = [record for record in records if record.sample_id not in completed_source_ids]
    return output_path, pending_records


def sanitize_existing_candidates(
    output_path: Path,
    *,
    valid_source_ids: set[str],
) -> set[str]:
    if not output_path.exists():
        return set()

    kept_lines: list[str] = []
    completed_source_ids: set[str] = set()
    rewrite_required = False
    with output_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                rewrite_required = True
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                rewrite_required = True
                continue
            source_sample_id = payload.get("source_sample_id")
            if source_sample_id not in valid_source_ids:
                rewrite_required = True
                continue
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                rewrite_required = True
                continue
            if metadata.get("prompt_version") != GENERATION_PROMPT_VERSION:
                rewrite_required = True
                continue
            if source_sample_id in completed_source_ids:
                rewrite_required = True
                continue
            kept_lines.append(line)
            completed_source_ids.add(str(source_sample_id))

    if rewrite_required:
        with output_path.open("w", encoding="utf-8") as handle:
            for line in kept_lines:
                handle.write(line + "\n")

    return completed_source_ids


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def validate_candidate(example: CandidateExample, image_count: int) -> bool:
    if len(example.messages) != 2:
        return False
    user_message, assistant_message = example.messages
    if user_message["role"] != "user" or assistant_message["role"] != "assistant":
        return False
    expected_prefix = "\n".join(f"<image_{index}>" for index in range(image_count))
    if not user_message["content"].startswith(expected_prefix):
        return False
    assistant_text = normalize_text(assistant_message["content"])
    if len(assistant_text) < 12:
        return False
    return not any(phrase in assistant_text for phrase in BANNED_META_PHRASES)


def load_candidate_examples(loader: BaseLoader) -> list[CandidateExample]:
    candidates: list[CandidateExample] = []
    for path in sorted(loader_candidates_dir(loader).glob("candidates_task*.jsonl")):
        candidates.extend(load_candidate_examples_from_path(path))
    return candidates


def load_candidate_examples_from_path(path: Path) -> list[CandidateExample]:
    candidates: list[CandidateExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            candidates.append(
                CandidateExample(
                    sample_id=payload["sample_id"],
                    source_sample_id=payload["source_sample_id"],
                    source_split=payload["source_split"],
                    source_row_index=int(payload["source_row_index"]),
                    task_type=payload["task_type"],
                    messages=payload["messages"],
                    metadata=payload["metadata"],
                )
            )
    return candidates


def make_parquet_rows(
    examples: list[CuratedExample],
    *,
    manifest_index: dict[tuple[str, str, int], ManifestRecord],
    loader: BaseLoader,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_cache: dict[tuple[str, str, int], SourceSample] = {}
    for example in examples:
        key = (example.source_sample_id, example.source_split, example.source_row_index)
        sample = source_cache.get(key)
        if sample is None:
            sample = loader.get_source_sample(
                manifest_index.get(
                    key,
                    ManifestRecord(
                        sample_id=example.source_sample_id,
                        split=example.source_split,
                        row_index=example.source_row_index,
                    ),
                )
            )
            source_cache[key] = sample
        rows.append(
            {
                "sample_id": example.sample_id,
                "source_sample_id": example.source_sample_id,
                "task_type": example.task_type,
                "images": [image.data for image in sample.images],
                "image_media_types": [image.media_type for image in sample.images],
                "messages": example.messages,
                "quality_score": example.quality_score,
                "metadata": {
                    "source_dataset": example.metadata.get("source_dataset") or "",
                    "source_split": example.metadata.get("source_split") or "",
                    "source_doc_id": example.metadata.get("source_doc_id") or "",
                    "data_source": example.metadata.get("data_source") or "",
                    "license": example.metadata.get("license") or "",
                    "url": example.metadata.get("url") or "",
                    "source_fields_json": example.metadata.get("source_fields_json") or "{}",
                    "generator_model": example.metadata.get("generator_model") or "",
                    "prompt_version": example.metadata.get("prompt_version") or "",
                    "judge_model": example.metadata.get("judge_model") or "",
                    "judge_prompt_version": example.metadata.get("judge_prompt_version") or "",
                    "created_at_utc": example.metadata.get("created_at_utc") or "",
                },
            }
        )
    return rows


def parquet_schema() -> pa.Schema:
    return pa.schema(
        [
            ("sample_id", pa.string()),
            ("source_sample_id", pa.string()),
            ("task_type", pa.string()),
            ("images", pa.list_(pa.binary())),
            ("image_media_types", pa.list_(pa.string())),
            (
                "messages",
                pa.list_(
                    pa.struct(
                        [
                            ("role", pa.string()),
                            ("content", pa.string()),
                        ]
                    )
                ),
            ),
            ("quality_score", pa.float32()),
            (
                "metadata",
                pa.struct(
                    [
                        ("source_dataset", pa.string()),
                        ("source_split", pa.string()),
                        ("source_doc_id", pa.string()),
                        ("data_source", pa.string()),
                        ("license", pa.string()),
                        ("url", pa.string()),
                        ("source_fields_json", pa.string()),
                        ("generator_model", pa.string()),
                        ("prompt_version", pa.string()),
                        ("judge_model", pa.string()),
                        ("judge_prompt_version", pa.string()),
                        ("created_at_utc", pa.string()),
                    ]
                ),
            ),
        ]
    )


def make_judge_messages(sample: SourceSample, example: CandidateExample) -> list[dict[str, object]]:
    content: list[dict[str, object]] = []
    for image in sample.images:
        content.append({"type": "image_url", "image_url": {"url": to_data_url(image)}})
    user_text = example.messages[0]["content"]
    assistant_text = example.messages[1]["content"]
    content.append(
        {
            "type": "text",
            "text": build_judge_instruction(
                user_text=user_text,
                assistant_text=assistant_text,
                task_type=example.task_type,
            ),
        }
    )
    return [{"role": "user", "content": content}]


def judge_examples(
    loader: BaseLoader,
    engine: ChatEngine,
    *,
    batch_size: int,
    model_repo: str,
    max_per_source_sample: int = DEFAULT_CANDIDATES_PER_SAMPLE,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 240,
) -> Path:
    raw_candidates = load_candidate_examples(loader)
    manifest_index = load_manifest_index(loader)
    unique: dict[tuple[str, str, str], CandidateExample] = {}
    by_source: dict[str, list[tuple[CandidateExample, float]]] = defaultdict(list)

    for candidate in raw_candidates:
        record = manifest_index.get(
            (
                candidate.source_sample_id,
                candidate.source_split,
                candidate.source_row_index,
            ),
            ManifestRecord(
                sample_id=candidate.source_sample_id,
                split=candidate.source_split,
                row_index=candidate.source_row_index,
            ),
        )
        source_sample = loader.get_source_sample(record)
        if not validate_candidate(candidate, len(source_sample.images)):
            continue
        key = (
            candidate.task_type,
            normalize_text(candidate.messages[0]["content"]),
            normalize_text(candidate.messages[1]["content"]),
        )
        unique.setdefault(key, candidate)

    unique_candidates = list(unique.values())
    source_cache: dict[tuple[str, str, int], SourceSample] = {}
    for batch_start in range(0, len(unique_candidates), batch_size):
        batch = unique_candidates[batch_start : batch_start + batch_size]
        source_samples: list[SourceSample] = []
        conversations: list[list[dict[str, object]]] = []
        for candidate in batch:
            record_key = (
                candidate.source_sample_id,
                candidate.source_split,
                candidate.source_row_index,
            )
            sample = source_cache.get(record_key)
            if sample is None:
                sample = loader.get_source_sample(
                    manifest_index.get(
                        record_key,
                        ManifestRecord(
                            sample_id=candidate.source_sample_id,
                            split=candidate.source_split,
                            row_index=candidate.source_row_index,
                        ),
                    )
                )
                source_cache[record_key] = sample
            source_samples.append(sample)
            conversations.append(make_judge_messages(sample, candidate))
        outputs = engine.chat(
            conversations,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        for candidate, output_text in zip(batch, outputs):
            payload = extract_json_object(output_text)
            jsonschema.validate(payload, JUDGE_SCHEMA)
            groundedness = float(payload["groundedness"])
            specificity = float(payload["specificity"])
            quality = float(payload["quality"])
            keep = bool(payload["keep"])
            score = (groundedness + specificity + quality) / 3.0
            if keep:
                by_source[candidate.source_sample_id].append((candidate, score))

    curated_dir = loader_curated_dir(loader)
    curated_dir.mkdir(parents=True, exist_ok=True)
    output_path = curated_dir / "curated.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for source_sample_id in sorted(by_source):
            selected = sorted(
                by_source[source_sample_id],
                key=lambda item: item[1],
                reverse=True,
            )[:max_per_source_sample]
            for candidate, score in selected:
                metadata = dict(candidate.metadata)
                metadata["judge_model"] = model_repo
                metadata["judge_prompt_version"] = JUDGE_PROMPT_VERSION
                curated = CuratedExample(
                    sample_id=candidate.sample_id,
                    source_sample_id=candidate.source_sample_id,
                    source_split=candidate.source_split,
                    source_row_index=candidate.source_row_index,
                    task_type=candidate.task_type,
                    messages=candidate.messages,
                    quality_score=score,
                    metadata=metadata,
                )
                handle.write(json.dumps(curated.to_dict(), ensure_ascii=False) + "\n")
    return output_path


def load_curated_examples(loader: BaseLoader) -> list[CuratedExample]:
    path = loader_curated_dir(loader) / "curated.jsonl"
    examples: list[CuratedExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            examples.append(
                CuratedExample(
                    sample_id=payload["sample_id"],
                    source_sample_id=payload["source_sample_id"],
                    source_split=payload["source_split"],
                    source_row_index=int(payload["source_row_index"]),
                    task_type=payload["task_type"],
                    messages=payload["messages"],
                    quality_score=float(payload["quality_score"]),
                    metadata=payload["metadata"],
                )
            )
    return examples


def load_export_examples(
    loader: BaseLoader,
    *,
    stage: str,
) -> list[CuratedExample]:
    if stage == "curated":
        return load_curated_examples(loader)
    if stage != "candidates":
        raise ValueError(f"Unsupported export stage: {stage!r}")

    examples: list[CuratedExample] = []
    for candidate in load_candidate_examples(loader):
        examples.append(
            CuratedExample(
                sample_id=candidate.sample_id,
                source_sample_id=candidate.source_sample_id,
                source_split=candidate.source_split,
                source_row_index=candidate.source_row_index,
                task_type=candidate.task_type,
                messages=candidate.messages,
                quality_score=0.0,
                metadata=candidate.metadata,
            )
        )
    return examples


def export_hf_dataset(loader: BaseLoader, *, stage: str = "candidates") -> Path:
    manifest_index = load_manifest_index(loader)
    output_dir = loader_export_dir(loader)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    source_sample_ids: set[str] = set()
    task_types: set[str] = set()

    if stage == "candidates":
        for path in sorted(loader_candidates_dir(loader).glob("candidates_task*.jsonl")):
            examples = [
                CuratedExample(
                    sample_id=candidate.sample_id,
                    source_sample_id=candidate.source_sample_id,
                    source_split=candidate.source_split,
                    source_row_index=candidate.source_row_index,
                    task_type=candidate.task_type,
                    messages=candidate.messages,
                    quality_score=0.0,
                    metadata=candidate.metadata,
                )
                for candidate in load_candidate_examples_from_path(path)
            ]
            rows = make_parquet_rows(
                examples,
                manifest_index=manifest_index,
                loader=loader,
            )
            if not rows:
                continue
            table = pa.Table.from_pylist(rows, schema=parquet_schema())
            pq.write_table(
                table,
                output_dir / f"{path.stem}.parquet",
                compression="zstd",
            )
            total_rows += len(rows)
            source_sample_ids.update(row["source_sample_id"] for row in rows)
            task_types.update(row["task_type"] for row in rows)
    else:
        rows = make_parquet_rows(
            load_export_examples(loader, stage=stage),
            manifest_index=manifest_index,
            loader=loader,
        )
        table = pa.Table.from_pylist(rows, schema=parquet_schema())
        pq.write_table(table, output_dir / f"{stage}.parquet", compression="zstd")
        total_rows = len(rows)
        source_sample_ids.update(row["source_sample_id"] for row in rows)
        task_types.update(row["task_type"] for row in rows)

    summary = {
        "rows": total_rows,
        "parquet_files": len(list(output_dir.glob("*.parquet"))),
        "source_samples": len(source_sample_ids),
        "task_types": sorted(task_types),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_dir


def set_cpu_affinity(worker_index: int) -> None:
    start = worker_index * 72
    stop = start + 72
    try:
        os.sched_setaffinity(0, range(start, stop))
    except AttributeError:
        return
