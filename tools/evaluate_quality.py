#!/usr/bin/env python3
"""
Evaluates sampled datasets for prompt leakage and pretraining data quality
using a vLLM-powered LLM.

Usage:
    # Interactive single-dataset test:
    python evaluate_quality.py --dataset hf___anthracite-org___pixmo-cap-images___processed

    # Full evaluation of all datasets:
    python evaluate_quality.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = (
    "You are an expert data quality analyst specializing in pretraining datasets for large multimodal models. "
    "Your analysis should be rigorous, specific, and actionable. Always back up claims with evidence from the sample. "
    "Write your response in clear markdown."
)

SINGLE_DATASET_PROMPT_TEMPLATE = """\
Dataset name: {dataset_name}

You are evaluating a **sample** of this dataset. The sample below consists of text excerpts and JSON metadata randomly drawn from a few shards. Because this is a small sample, base your analysis on what is visible, and explicitly note where sample size limits certainty.

Please produce a structured markdown report with the following sections:

## 1. Prompt Leak Detection
Scan all provided text for any obvious undesirable artifacts, system prompts, or prompt leaks. These are fragments of instructions that were likely given to a data generation or formatting system and accidentally left in the training data. Examples of the KIND of thing to look for include template placeholders, meta-instructions (e.g., "Rewrite the following", "As an AI assistant", "Summarize the following"), or synthetic-sounding directives like "This well-structured sentence integrates...".

**CRITICAL INSTRUCTION**: Only report leaks that are EXACTLY present in the Sample Data below. Do NOT mention examples, hypothetical leaks, or anything not found in the sample. It is better to report nothing than to hallucinate a leak. If no leaks are present, write exactly: "No prompt leaks detected in the provided sample." and nothing else in this section.

For each actual finding, provide:
- The exact snippet (in backticks).
- A severity rating: Critical / High / Medium / Low.
- A brief explanation.

## 2. Pretraining Data Quality Analysis
Evaluate the sample for its suitability as pretraining data. Address the following dimensions:

### 2a. Diversity
Does the content cover a variety of topics, styles, and structures? Or does it feel narrow and repetitive? Comment on language diversity as well.

### 2b. Duplicates
Are there near-duplicate or exact-duplicate entries within this sample? Note any repeated phrases, templates, or structural clones.

### 2c. Balance
If applicable, is there a reasonable balance across different classes, domains, or concepts? For vision datasets with captions, does the sample suggest a good variety of visual concepts?

### 2d. Potential Harm
Could this data harm a model's training process or downstream behavior? Consider:
- Toxic, biased, or NSFW content.
- Excessive repetition that could cause memorization or mode collapse.
- Misinformation or low-quality text.
- Formatting artifacts (e.g., HTML tags, markdown, excessive whitespace) that could leak into generation behavior.
- Any content that seems auto-generated in a low-quality way.

## 3. Overall Assessment
Provide an overall quality score from 1 (unusable) to 10 (excellent) for pretraining suitability, with a concise justification.

---

### Sample Data
{sample_text}
"""

FINAL_SUMMARY_PROMPT_TEMPLATE = """\
You are the lead data quality reviewer. Below is **structured data** extracted from individual quality reports for {num_datasets} datasets, followed by the full reports for reference.

**STRUCTURED DATA (use this as ground truth for scores and severities):**
{structured_table}

The final report must contain:
1. **Executive Summary**: 2-3 sentences summarizing the overall health of the dataset collection.
2. **Per-Dataset Scores**: A markdown table with columns: Dataset Name, Overall Score (1-10), Prompt Leak Severity (None / Low / Medium / High / Critical), Key Finding. Use the Structured Data above for exact values.
3. **Prompt Leakage Statistics**: Aggregate counts of how many datasets had leaks, by severity.
4. **Flagged Datasets**: A list of any datasets that should be immediately removed or require urgent investigation, with reasons.
5. **General Recommendations**: Actionable advice for improving the pretraining data pipeline based on these findings.

Here are the full individual reports for context:
{all_reports_text}
"""


def extract_strings_from_json(obj, max_len=2000):
    """Recursively extract strings from a JSON object, skipping long base64-like strings."""
    strings = []
    if isinstance(obj, str):
        if len(obj) > max_len and all(
            c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in obj
        ):
            strings.append("[BASE64_DATA_TRUNCATED]")
        else:
            strings.append(obj)
    elif isinstance(obj, list):
        for item in obj:
            strings.extend(extract_strings_from_json(item, max_len))
    elif isinstance(obj, dict):
        for v in obj.values():
            strings.extend(extract_strings_from_json(v, max_len))
    return strings


def parse_report(report_text: str) -> dict:
    """Parse an individual report to extract score, leak severity, and key finding."""
    # Extract score
    score = "N/A"
    score_match = re.search(
        r"(?:Quality Score|Overall quality score)[:/]?\s*(?:\*\*)?(\d+)",
        report_text,
        re.IGNORECASE,
    )
    if score_match:
        score = score_match.group(1).strip()

    # Extract leak severity from Section 1
    leak_severity = "None"
    leak_section_match = re.search(
        r"## 1\. Prompt Leak Detection(.*?)(?=## 2\.|\Z)",
        report_text,
        re.DOTALL | re.IGNORECASE,
    )
    if leak_section_match:
        leak_text = leak_section_match.group(1)
        if "No prompt leaks detected" in leak_text:
            leak_severity = "None"
        else:
            for sev in ["Critical", "High", "Medium", "Low"]:
                if sev in leak_text:
                    leak_severity = sev
                    break
    else:
        leak_severity = "None"

    # Extract key finding: justification or first sentence of overall assessment
    key_finding = ""
    justification_match = re.search(
        r"\*\*Justification:\*\*\s*(.+?)(?:\n\n|\Z)",
        report_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not justification_match:
        justification_match = re.search(r"Justification:\s*(.+?)(?:\n\n|\Z)", report_text, re.DOTALL | re.IGNORECASE)
    if justification_match:
        key_finding = justification_match.group(1).strip().split("\n")[0].strip()
    else:
        overall_match = re.search(
            r"## 3\. Overall Assessment\s*(.+?)(?:\n\n|\Z)",
            report_text,
            re.DOTALL | re.IGNORECASE,
        )
        if overall_match:
            key_finding = overall_match.group(1).strip().split("\n")[0].strip()

    if len(key_finding) > 200:
        key_finding = key_finding[:197] + "..."

    return {
        "score": score,
        "leak_severity": leak_severity,
        "key_finding": key_finding,
    }


def collect_sample_text(dataset_path: Path, max_chars: int = 30_000) -> str:
    """Collect text sample from a dataset directory."""
    text_parts = []
    current_len = 0

    for file_path in sorted(dataset_path.rglob("*")):
        if current_len >= max_chars:
            break
        if file_path.is_dir():
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            header = f"\n--- FILE: {file_path.relative_to(dataset_path)} ---\n"
            part = header + content
            text_parts.append(part)
            current_len += len(part)

        elif suffix == ".json":
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                strings = extract_strings_from_json(data)
                content = "\n".join(strings)
            except Exception:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
            header = f"\n--- FILE: {file_path.relative_to(dataset_path)} (JSON text extraction) ---\n"
            part = header + content
            text_parts.append(part)
            current_len += len(part)

        # Skip images and other binary files

    full_text = "\n".join(text_parts)
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n[...SAMPLE TRUNCATED...]"
    return full_text


def format_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_reports(llm, tokenizer, sampling_params, datasets, args):
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    metadata = {}

    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"{'=' * 60}")

        sample_text = collect_sample_text(dataset_path, max_chars=args.max_chars_per_dataset)

        if not sample_text.strip():
            print(f"Warning: No text found for dataset {dataset_name}. Skipping.")
            continue

        user_prompt = SINGLE_DATASET_PROMPT_TEMPLATE.format(
            dataset_name=dataset_name,
            sample_text=sample_text,
        )

        prompt = format_chat_prompt(tokenizer, SYSTEM_PROMPT, user_prompt)

        outputs = llm.generate([prompt], sampling_params)
        report = outputs[0].outputs[0].text.strip()

        report_path = reports_dir / f"{dataset_name}_report.md"
        report_path.write_text(report, encoding="utf-8")
        reports[dataset_name] = report
        metadata[dataset_name] = parse_report(report)
        print(f"Saved report to {report_path}")

    return reports, metadata


def generate_final_summary(llm, tokenizer, sampling_params, reports, metadata, args):
    print(f"\n{'=' * 60}")
    print("Generating final summary report...")
    print(f"{'=' * 60}")

    # Build structured table from parsed metadata to prevent hallucination
    rows = []
    for dataset_name in sorted(reports.keys()):
        meta = metadata[dataset_name]
        rows.append(f"| {dataset_name} | {meta['score']} | {meta['leak_severity']} | {meta['key_finding']} |")
    structured_table = (
        "| Dataset Name | Overall Score | Prompt Leak Severity | Key Finding |\n|---|---|---|---|\n" + "\n".join(rows)
    )

    all_reports_text = ""
    for dataset_name, report in reports.items():
        all_reports_text += (
            f"\n\n--- BEGIN REPORT: {dataset_name} ---\n\n{report}\n\n--- END REPORT: {dataset_name} ---\n"
        )

    # Conservative character-based truncation to avoid exceeding context window.
    # Assuming ~4 chars per token on average, 120k chars ~ 30k tokens.
    max_context_chars = 120_000
    if len(all_reports_text) > max_context_chars:
        all_reports_text = all_reports_text[:max_context_chars] + "\n[...TRUNCATED...]"

    user_prompt = FINAL_SUMMARY_PROMPT_TEMPLATE.format(
        num_datasets=len(reports),
        structured_table=structured_table,
        all_reports_text=all_reports_text,
    )

    prompt = format_chat_prompt(tokenizer, SYSTEM_PROMPT, user_prompt)

    outputs = llm.generate([prompt], sampling_params)
    summary = outputs[0].outputs[0].text.strip()

    summary_path = Path(args.final_report_path)
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Saved final summary to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate dataset quality using vLLM.")
    parser.add_argument(
        "--sampled-dir",
        type=str,
        default="../verify/sampled",
        help="Path to sampled datasets directory",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="./reports",
        help="Directory to save individual reports",
    )
    parser.add_argument(
        "--final-report-path",
        type=str,
        default="./overall_quality_report.md",
        help="Path for the final summary report",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.6-27B",
        help="vLLM model name or path",
    )
    parser.add_argument("--tp-size", type=int, default=4, help="Tensor parallel size (number of GPUs)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="If set, only evaluate this single dataset name",
    )
    parser.add_argument(
        "--max-chars-per-dataset",
        type=int,
        default=30_000,
        help="Max characters of sample text per dataset",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens to generate per LLM call",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    args = parser.parse_args()

    sampled_dir = Path(args.sampled_dir)
    if not sampled_dir.exists():
        print(f"Error: Sampled directory {sampled_dir} does not exist.")
        sys.exit(1)

    # Discover datasets (top-level directories in sampled/)
    datasets = {item.name: item for item in sorted(sampled_dir.iterdir()) if item.is_dir()}

    if args.dataset:
        if args.dataset not in datasets:
            print(f"Error: Dataset '{args.dataset}' not found in {sampled_dir}.")
            print(f"Available datasets: {list(datasets.keys())}")
            sys.exit(1)
        datasets = {args.dataset: datasets[args.dataset]}

    if not datasets:
        print("No datasets found.")
        sys.exit(1)

    print(f"Will evaluate {len(datasets)} dataset(s): {list(datasets.keys())}")
    print(f"Loading model: {args.model} with tensor_parallel_size={args.tp_size}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        dtype="bfloat16",
        gdn_prefill_backend="triton",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.95,
        repetition_penalty=1.05,
    )

    reports, metadata = generate_reports(llm, tokenizer, sampling_params, datasets, args)

    if not args.dataset:
        generate_final_summary(llm, tokenizer, sampling_params, reports, metadata, args)
    else:
        print("\nSingle-dataset mode: skipping final summary.")

    print("\nDone.")


if __name__ == "__main__":
    main()
