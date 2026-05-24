"""WebDataset (tar-shard) export for the google_rsrcc loader.

WebDataset format
-----------------
Each sample in a shard tar file is three consecutive entries sharing the same key:

  {key}.before.png   — before image bytes (PNG)
  {key}.after.png    — after image bytes (PNG)
  {key}.json         — JSON with all fields needed for training

The JSON schema per sample::

    {
      "sample_id":         str,
      "source_sample_id":  str,
      "source_split":      str,          # "train" | "val"
      "question":          str,          # verbatim from source CSV
      "answer":            str,          # verbatim from source CSV
      "reasoning":         str,          # generated CoT (no <think> tags)
      "messages": [
        {"role": "user",      "content": "<image_0>\\n<image_1>\\n{question}"},
        {"role": "assistant", "content": "<think>{reasoning}</think>\\n{answer}"}
      ],
      "metadata": { generator_model, prompt_version, created_at_utc, ... }
    }

The ``messages`` field is SFT-ready: training code substitutes ``<image_0>`` /
``<image_1>`` with the images stored in the same tar entry.

One output shard tar is produced per input ``candidates_task*.jsonl`` file,
preserving the sharding structure for distributed training data loading.

Usage (standalone script, see ``scripts/export_rsrcc_wds.py``)::

    from sft_recaption.wds_export import export_rsrcc_wds
    from sft_recaption.loaders import create_loader

    loader = create_loader("google_rsrcc")
    summary = export_rsrcc_wds(loader)
"""

from __future__ import annotations

import io
import json
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sft_recaption.config import CANDIDATES_DIR
from sft_recaption.loaders.base import BaseLoader

# ------------------------------------------------------------------ #
# Low-level tar writer
# ------------------------------------------------------------------ #


def _add_tar_entry(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = int(time.time())
    tf.addfile(info, io.BytesIO(data))


# ------------------------------------------------------------------ #
# Per-candidate processing
# ------------------------------------------------------------------ #


def _candidate_to_wds_sample(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a raw JSONL candidate dict to a WebDataset sample dict.

    Returns None if the candidate is malformed or missing required fields.
    """
    messages = candidate.get("messages")
    metadata = candidate.get("metadata") or {}

    if not isinstance(messages, list) or len(messages) != 2:
        return None

    reasoning = messages[1].get("content", "").strip()
    if not reasoning:
        return None

    source_fields_raw = metadata.get("source_fields_json", "{}")
    try:
        source_fields: dict = json.loads(source_fields_raw)
    except json.JSONDecodeError:
        return None

    question = source_fields.get("question", "")
    answer = source_fields.get("answer", "")
    before_path = source_fields.get("before_path", "")
    after_path = source_fields.get("after_path", "")

    if not (question and answer and before_path and after_path):
        return None

    # Assemble SFT-ready assistant message: <think>{reasoning}</think>\n{answer}
    assistant_content = f"<think>{reasoning}</think>\n{answer}"
    user_content = f"<image_0>\n<image_1>\n{question}"

    return {
        "sample_id": candidate.get("sample_id", ""),
        "source_sample_id": candidate.get("source_sample_id", ""),
        "source_split": candidate.get("source_split", ""),
        "question": question,
        "answer": answer,
        "reasoning": reasoning,
        "before_path": before_path,
        "after_path": after_path,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {k: v for k, v in metadata.items() if k != "source_fields_json"},
    }


# ------------------------------------------------------------------ #
# Shard writer
# ------------------------------------------------------------------ #


def _write_shard(
    candidate_path: Path,
    output_path: Path,
) -> dict[str, int]:
    """Write one WebDataset tar shard from a candidates JSONL file.

    Returns a stats dict with ``{written, skipped}`` counts.
    """
    written = 0
    skipped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(output_path, mode="w:gz") as tf:
        with candidate_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                sample = _candidate_to_wds_sample(candidate)
                if sample is None:
                    skipped += 1
                    continue

                before_path = Path(sample.pop("before_path"))
                after_path = Path(sample.pop("after_path"))

                try:
                    before_bytes = before_path.read_bytes()
                    after_bytes = after_path.read_bytes()
                except OSError:
                    skipped += 1
                    continue

                # Sanitise key: use source_sample_id stripped of path-unsafe chars
                raw_key = sample["source_sample_id"]
                key = (
                    "".join(c if (c.isalnum() or c in {"-", "_"}) else "_" for c in raw_key).strip("_")
                    or f"sample_{written:08d}"
                )

                json_bytes = json.dumps(sample, ensure_ascii=False).encode("utf-8")

                _add_tar_entry(tf, f"{key}.before.png", before_bytes)
                _add_tar_entry(tf, f"{key}.after.png", after_bytes)
                _add_tar_entry(tf, f"{key}.json", json_bytes)
                written += 1

    if written == 0:
        # Remove empty shard
        output_path.unlink(missing_ok=True)

    return {"written": written, "skipped": skipped}


# ------------------------------------------------------------------ #
# Main export entry point
# ------------------------------------------------------------------ #


def export_rsrcc_wds(
    loader: BaseLoader,
    *,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Export all RSRCC candidates to WebDataset tar shards.

    One ``.tar.gz`` shard is written per ``candidates_task*.jsonl`` file.
    Shards with zero written samples are deleted.

    Args:
        loader: A ``GoogleRsrccLoader`` instance.
        output_dir: Destination directory (defaults to
            ``artifacts/wds/<loader.name>/``).
        verbose: Print per-shard progress.

    Returns:
        Summary dict with ``{shards_written, total_written, total_skipped,
        output_dir, generated_at_utc}``.
    """
    if output_dir is None:
        output_dir = CANDIDATES_DIR.parent / "wds" / loader.name

    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_dir = CANDIDATES_DIR / loader.name
    candidate_paths = sorted(candidates_dir.glob("candidates_task*.jsonl"))

    if not candidate_paths:
        raise FileNotFoundError(
            f"No candidates_task*.jsonl files found under {candidates_dir}. Run the generate step first."
        )

    total_written = 0
    total_skipped = 0
    shards_written = 0

    for candidate_path in candidate_paths:
        # e.g. candidates_task0042.jsonl → shard_0042.tar.gz
        task_suffix = candidate_path.stem.replace("candidates_", "")  # task0042
        shard_name = f"shard_{task_suffix}.tar.gz"
        output_path = output_dir / shard_name

        stats = _write_shard(candidate_path, output_path)
        total_written += stats["written"]
        total_skipped += stats["skipped"]
        if stats["written"] > 0:
            shards_written += 1

        if verbose:
            status = "✓" if stats["written"] > 0 else "∅"
            print(f"  {status}  {shard_name}: {stats['written']} written, {stats['skipped']} skipped")

    summary = {
        "shards_written": shards_written,
        "total_written": total_written,
        "total_skipped": total_skipped,
        "output_dir": str(output_dir),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
