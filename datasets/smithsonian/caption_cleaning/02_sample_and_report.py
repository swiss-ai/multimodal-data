#!/usr/bin/env python3
"""
02_sample_and_report.py

For each of the 25 WDS subset directories:
  1. Sample 8 captions uniformly across all shards.
  2. Pre-filter: mark captions with len < 120 as PRE_FILTERED.
  3. For CLEAN subsets: run LLM cleaning (Gemma 4 31B-it, text-only).
  4. Post-filter: mark cleaned captions with len < 120 as POST_FILTERED.
  5. Write a full before/after markdown report to data/cleaning_report.md.

SKIP subsets are shown in the report with raw samples only (no LLM call).
"""

import json
import random
import sys
import tarfile
from pathlib import Path

from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from prompts import PROMPT_BY_SUBSET  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WDS_ROOT = Path("/path/to/data/vision-datasets/processed/smithsonian")
REPORT_PATH = Path(__file__).parent.parent / "data" / "cleaning_report.md"

MODEL_PATH = "/tmp/models/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.80
MAX_TOKENS = 1024
TEMPERATURE = 0.1
TOP_P = 0.9

SAMPLES_PER_SUBSET = 8
MIN_LEN = 200  # pre- and post-filter threshold

RANDOM_SEED = 42
CHECKPOINT_PATH = Path(__file__).parent.parent / "data" / "cleaning_checkpoint.json"

# Subsets where LLM cleaning is worth doing (10/25).
CLEAN_SUBSETS = {
    "tier1/nmaahc",
    "tier1/nmafa",
    "tier1/npg",
    "tier1/saam",
    "tier2/design/chndm",
    "tier2/history/nasm",
    "tier2/history/nmah",
    "tier2/other/acm",
    "tier2/other/npm",
    "tier2/other/sia",
}

# Skip reasons for the report (for non-CLEAN subsets)
SKIP_REASONS: dict[str, str] = {
    "tier2/art/hmsg": "pure metadata only, nothing left after stripping",
    "tier2/history/aaa": 'synthetic templates ("Front view of a cassette tape...")',
    "tier2/other/nmai": 'all identical: "Native American artifact . Native american artifact."',
    "tier2/other/nzp": "metadata + keyword tag lists, no prose",
    "tier3/nmnh/anthro": "template (object + location only), no prose",
    "tier3/nmnh/birds": "template (species + taxonomy + location)",
    "tier3/nmnh/botany": "template (species + taxonomy + location)",
    "tier3/nmnh/ento": "template (species + taxonomy + notes)",
    "tier3/nmnh/fishes": "template (species + taxonomy + location)",
    "tier3/nmnh/herps": "template (species + taxonomy + location)",
    "tier3/nmnh/inv": "very short (20-113c), mostly just species + preservation",
    "tier3/nmnh/mammals": "template (species + taxonomy + location)",
    "tier3/nmnh/minsci": "very short (18-93c), mineral name + location only",
    "tier3/nmnh/paleo": "template (species + taxonomy + location + age)",
    "tier4/3d": 'all identical 47c "Rendered from a 3D scan of the museum specimen."',
}

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def find_tar_files(subset_dir: Path) -> list[Path]:
    """Return all .tar files directly in subset_dir (not in subdirs like 'sample/')."""
    return sorted(subset_dir.glob("*.tar"))


def sample_txt_from_tar(tar_path: Path, rng: random.Random, max_collect: int = 8) -> str | None:
    """
    Open a tar and return the text content of one randomly chosen .txt member.
    Reads only the first max_collect .txt entries (not all headers) for speed.
    Returns None if the tar contains no .txt members.
    """
    txts: list[str] = []
    try:
        with tarfile.open(tar_path, "r") as tf:
            for member in tf:
                if member.name.endswith(".txt") and member.isfile():
                    f = tf.extractfile(member)
                    if f is not None:
                        txts.append(f.read().decode("utf-8", errors="replace").strip())
                    if len(txts) >= max_collect:
                        break
    except Exception as exc:
        print(f"  WARNING: could not read {tar_path}: {exc}", file=sys.stderr)
    return rng.choice(txts) if txts else None


def sample_captions(subset_dir: Path, n: int, rng: random.Random) -> list[str]:
    """
    Sample up to n captions uniformly spread across all tars in subset_dir.
    Falls back to reading more from fewer tars if the subset is small.
    """
    tars = find_tar_files(subset_dir)
    if not tars:
        return []

    captions: list[str] = []
    attempts = 0
    max_attempts = n * 10

    # Choose tar indices spread across the full range, then cycle if needed
    if len(tars) >= n:
        indices = [int(i * len(tars) / n) for i in range(n)]
    else:
        # Fewer tars than samples needed; cycle through tars multiple times
        indices = [i % len(tars) for i in range(n)]

    chosen_tars = [tars[i] for i in indices]
    rng.shuffle(chosen_tars)

    for tar_path in chosen_tars:
        if len(captions) >= n or attempts >= max_attempts:
            break
        txt = sample_txt_from_tar(tar_path, rng)
        attempts += 1
        if txt and txt not in captions:
            captions.append(txt)

    # If still short, try remaining tars in random order
    remaining = [t for t in tars if t not in set(chosen_tars)]
    rng.shuffle(remaining)
    for tar_path in remaining:
        if len(captions) >= n or attempts >= max_attempts:
            break
        txt = sample_txt_from_tar(tar_path, rng)
        attempts += 1
        if txt and txt not in captions:
            captions.append(txt)

    return captions[:n]


# ---------------------------------------------------------------------------
# Discover all 25 subset directories
# ---------------------------------------------------------------------------


def discover_subsets() -> list[tuple[str, Path]]:
    """
    Return list of (subset_key, subset_dir) for every leaf directory that
    contains at least one .tar file.
    """
    results = []
    for path in sorted(WDS_ROOT.rglob("*.tar")):
        subset_dir = path.parent
        # Skip 'sample' subdirs
        if subset_dir.name == "sample":
            continue
        subset_key = str(subset_dir.relative_to(WDS_ROOT))
        entry = (subset_key, subset_dir)
        if entry not in results:
            results.append(entry)
    return results


# ---------------------------------------------------------------------------
# LLM cleaning
# ---------------------------------------------------------------------------


def build_messages(subset_key: str, caption: str) -> list[dict]:
    prompt_template = PROMPT_BY_SUBSET.get(subset_key, PROMPT_BY_SUBSET["default"])
    prompt = prompt_template.format(caption=caption)
    return [{"role": "user", "content": prompt}]


def clean_captions(
    llm: LLM,
    sampling: SamplingParams,
    items: list[tuple[str, str, str]],  # (subset_key, sample_id, caption)
) -> dict[str, str]:
    """
    Run LLM cleaning on all items in one batch.
    Returns dict mapping sample_id -> cleaned_text.
    """
    conversations = [build_messages(subset, cap) for subset, _, cap in items]
    outputs = llm.chat(conversations, sampling_params=sampling, use_tqdm=True)
    return {sample_id: output.outputs[0].text.strip() for (_, sample_id, _), output in zip(items, outputs)}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(subset_results: list[dict]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(subset_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")


def load_checkpoint() -> list[dict] | None:
    if not CHECKPOINT_PATH.exists():
        return None
    data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    print(f"Loaded checkpoint from: {CHECKPOINT_PATH}")
    return data


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(
    report_path: Path,
    subset_results: list[dict],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sep = "─" * 72
    lines = ["# Smithsonian Caption Cleaning Report\n"]

    for result in subset_results:
        subset_key = result["subset"]
        is_clean = result["is_clean"]
        samples = result["samples"]

        if is_clean:
            n_pre = sum(1 for s in samples if s["status"] == "PRE_FILTERED")
            n_cleaned = sum(1 for s in samples if s["status"] == "CLEANED")
            n_post = sum(1 for s in samples if s["status"] == "POST_FILTERED")
            lines.append(
                f"## {subset_key}  CLEAN  ({n_pre} pre-filtered, {n_cleaned} cleaned, {n_post} post-filtered)\n"
            )
            for i, s in enumerate(samples, 1):
                status = s["status"]
                raw = s["raw"]
                cleaned = s["cleaned"]
                after_len = f" → {len(cleaned)}c" if cleaned else ""
                lines.append(f"### [{i}] {status}  ({len(raw)}c{after_len})\n")
                if status == "PRE_FILTERED":
                    lines.append(f"```\n{raw}\n```\n")
                else:
                    lines.append(f"**before** ({len(raw)}c):\n```\n{raw}\n```\n")
                    lines.append(f"**after** ({len(cleaned)}c):\n```\n{cleaned}\n```\n")
                lines.append(sep + "\n")
        else:
            skip_reason = SKIP_REASONS.get(subset_key, "no useful prose")
            lines.append(f"## {subset_key}  SKIP — {skip_reason}\n")
            for i, s in enumerate(samples, 1):
                raw = s["raw"]
                lines.append(f"### [{i}] ({len(raw)}c)\n```\n{raw}\n```\n")
                lines.append(sep + "\n")

        lines.append("\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # If checkpoint exists, skip straight to report generation
    cached = load_checkpoint()
    if cached is not None:
        write_report(REPORT_PATH, cached)
        print("\n=== Summary (from checkpoint) ===")
        for result in cached:
            subset_key = result["subset"]
            samples = result["samples"]
            if result["is_clean"]:
                n_pre = sum(1 for s in samples if s["status"] == "PRE_FILTERED")
                n_cleaned = sum(1 for s in samples if s["status"] == "CLEANED")
                n_post = sum(1 for s in samples if s["status"] == "POST_FILTERED")
                print(f"  {subset_key}: {n_cleaned} cleaned, {n_pre} pre-filtered, {n_post} post-filtered")
            else:
                print(f"  {subset_key}: SKIP")
        return

    rng = random.Random(RANDOM_SEED)

    print("Discovering subset directories ...")
    subsets = discover_subsets()
    print(f"Found {len(subsets)} subsets.\n")

    # -----------------------------------------------------------------------
    # Step 1: Sample captions from all subsets
    # -----------------------------------------------------------------------
    subset_results: list[dict] = []
    llm_batch: list[tuple[str, str, str]] = []  # (subset_key, sample_id, caption)

    for subset_key, subset_dir in subsets:
        print(f"Sampling {subset_key} ...")
        raw_captions = sample_captions(subset_dir, SAMPLES_PER_SUBSET, rng)
        is_clean = subset_key in CLEAN_SUBSETS

        samples = []
        for i, raw in enumerate(raw_captions):
            sample_id = f"{subset_key}:{i}"
            if len(raw) < MIN_LEN:
                samples.append({"raw": raw, "cleaned": None, "status": "PRE_FILTERED"})
            elif is_clean:
                samples.append({"raw": raw, "cleaned": None, "status": "PENDING"})
                llm_batch.append((subset_key, sample_id, raw))
            else:
                # SKIP subset — no LLM call
                samples.append({"raw": raw, "cleaned": None, "status": "SKIP"})

        subset_results.append({"subset": subset_key, "is_clean": is_clean, "samples": samples})

    # -----------------------------------------------------------------------
    # Step 2: Run LLM on all CLEAN captions that passed pre-filter
    # -----------------------------------------------------------------------
    if llm_batch:
        print(f"\nLoading Gemma 4 31B-it (tensor_parallel_size={TENSOR_PARALLEL_SIZE}) ...")
        llm = LLM(
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            dtype="bfloat16",
            max_model_len=4096,
            trust_remote_code=True,
        )
        print(f"Running LLM on {len(llm_batch)} captions ...")
        sampling = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS)
        cleaned_map = clean_captions(llm, sampling, llm_batch)
    else:
        cleaned_map = {}

    # -----------------------------------------------------------------------
    # Step 3: Fill in cleaned text and apply post-filter
    # -----------------------------------------------------------------------
    for result in subset_results:
        subset_key = result["subset"]
        for i, s in enumerate(result["samples"]):
            if s["status"] != "PENDING":
                continue
            sample_id = f"{subset_key}:{i}"
            cleaned = cleaned_map.get(sample_id, "")
            s["cleaned"] = cleaned
            if len(cleaned) < MIN_LEN:
                s["status"] = "POST_FILTERED"
            else:
                s["status"] = "CLEANED"

    # -----------------------------------------------------------------------
    # Step 4: Save checkpoint + write report
    # -----------------------------------------------------------------------
    save_checkpoint(subset_results)
    write_report(REPORT_PATH, subset_results)

    # Print summary
    print("\n=== Summary ===")
    for result in subset_results:
        subset_key = result["subset"]
        samples = result["samples"]
        is_clean = result["is_clean"]
        if is_clean:
            n_pre = sum(1 for s in samples if s["status"] == "PRE_FILTERED")
            n_cleaned = sum(1 for s in samples if s["status"] == "CLEANED")
            n_post = sum(1 for s in samples if s["status"] == "POST_FILTERED")
            print(f"  {subset_key}: {n_cleaned} cleaned, {n_pre} pre-filtered, {n_post} post-filtered")
        else:
            print(f"  {subset_key}: SKIP")


if __name__ == "__main__":
    main()
