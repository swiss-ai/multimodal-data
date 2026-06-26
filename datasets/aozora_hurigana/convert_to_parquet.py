"""
Convert Aozora Hurigana Speech Corpus v2 (nested zips of mp3 + txt)
to HF-style parquet with audio bytes + transcription columns.

Processes inner author zips in parallel — never extracts to disk.
All mp3 bytes are read in memory and written directly to parquet.

Important: preserve the rendition/source-book directory in the output schema.
Different renditions of the same work reuse both ``line_num`` and mp3 names, so
flattening to only ``author/work`` destroys the sequence identity needed for
later interleave.

Usage:
    python convert_to_parquet.py \
        --input-dir /path/to/raw/ndl___aozora_hurigana_speech_v2 \
        --output-dir /path/to/raw/ndl___aozora_hurigana_speech_v2/parquet \
        --num-workers 256
"""
import argparse
import io
import re
import zipfile
from multiprocessing import Pool
from pathlib import Path

import polars as pl


def parse_annotation(txt_content: str, work_name: str):
    """Parse annotation txt → list of dicts with all metadata.

    Each entry in the txt looks like:
        行番号\tN\tNNNNNNN.mp3
        <whisper result>\t[音声認識結果_1]
        <aozora text>\t[青空文庫テキスト]
        「解析結果:」
        <kanji> <whisper_reading> <correct_reading> <error_type>
        「読み推定結果:」
        <kanji> <reading> <reading> <source>
    """
    entries = []
    lines = txt_content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r"^行番号\t(\d+)\t(\d+\.mp3)$", line)
        if m:
            line_num = int(m.group(1))
            mp3_name = m.group(2)
            asr_result = None
            aozora_text = None
            analysis = []
            reading_estimates = []

            j = i + 1
            # Parse ASR result
            while j < len(lines) and j < i + 20:
                l = lines[j].strip()
                if "[音声認識結果" in l:
                    asr_result = l.split("\t")[0].strip()
                    j += 1
                    continue
                if "[青空文庫テキスト]" in l:
                    raw = l.split("\t")[0].strip()
                    aozora_text = re.sub(r"^[!！]\s*", "", raw)
                    j += 1
                    continue
                if l == "「解析結果:」":
                    j += 1
                    while j < len(lines):
                        al = lines[j].strip()
                        if not al or al.startswith("「") or al.startswith("行番号"):
                            break
                        analysis.append(al)
                        j += 1
                    continue
                if l == "「読み推定結果:」":
                    j += 1
                    while j < len(lines):
                        rl = lines[j].strip()
                        if not rl or rl.startswith("「") or rl.startswith("行番号"):
                            break
                        reading_estimates.append(rl)
                        j += 1
                    continue
                if l.startswith("行番号\t"):
                    break
                j += 1

            if aozora_text:
                entries.append({
                    "mp3_name": mp3_name,
                    "line_num": line_num,
                    "transcription": aozora_text,
                    "asr_result": asr_result or "",
                    "work": work_name,
                    "analysis": "\n".join(analysis) if analysis else "",
                    "reading_estimates": "\n".join(reading_estimates) if reading_estimates else "",
                })
            i = j
        else:
            i += 1
    return entries


def make_source_id(author_name: str, work_name: str, rendition_name: str) -> str:
    """Build a slash-free source ID safe for later universal cut IDs."""
    return f"{author_name}__{work_name}__{rendition_name}"


def validate_rendition_entries(entries, txt_path: str):
    """Assert the original annotation is sequential within one rendition."""
    line_nums = [e["line_num"] for e in entries]
    mp3_names = [e["mp3_name"] for e in entries]
    if len(line_nums) != len(set(line_nums)):
        raise RuntimeError(f"Duplicate line_num detected in rendition annotation: {txt_path}")
    if len(mp3_names) != len(set(mp3_names)):
        raise RuntimeError(f"Duplicate mp3_name detected in rendition annotation: {txt_path}")


def decode_zip_filename(name: str) -> str:
    """Decode Shift-JIS filenames that were read as CP437 by Python's zipfile."""
    try:
        return name.encode("cp437").decode("shift_jis")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name


def process_one_author(args):
    """Process a single inner author zip from the outer zip. Returns list of dicts."""
    outer_zip_path, inner_name = args
    author_name = decode_zip_filename(Path(inner_name).stem)

    with zipfile.ZipFile(outer_zip_path) as outer:
        inner_data = io.BytesIO(outer.read(inner_name))

    try:
        inner = zipfile.ZipFile(inner_data)
    except zipfile.BadZipFile:
        return []

    names_set = set(inner.namelist())
    txt_files = [n for n in names_set if n.endswith(".txt") and "csv" not in n.lower()]

    rows = []
    for txt_path in txt_files:
        try:
            content = inner.read(txt_path).decode("utf-8", errors="ignore")
        except Exception:
            continue

        decoded_txt_path = Path(decode_zip_filename(txt_path))
        if len(decoded_txt_path.parts) < 3:
            continue

        work_name = decoded_txt_path.parts[0]
        rendition_name = decoded_txt_path.parts[1]
        entries = parse_annotation(content, work_name)
        if not entries:
            continue

        validate_rendition_entries(entries, str(decoded_txt_path))

        mp3_dir = str(Path(txt_path).parent) + "/mp3/"
        source_id = make_source_id(author_name, work_name, rendition_name)

        for entry in entries:
            mp3_path = mp3_dir + entry["mp3_name"]
            if mp3_path not in names_set:
                continue
            try:
                audio_bytes = inner.read(mp3_path)
                clip_num = int(entry["line_num"])
                rows.append({
                    "sample_id": f"{source_id}_{clip_num:06d}",
                    "source_id": source_id,
                    "clip_num": clip_num,
                    "audio": {
                        "bytes": audio_bytes,
                        "path": f"{author_name}/{work_name}/{rendition_name}/mp3/{entry['mp3_name']}",
                    },
                    "transcription": entry["transcription"],
                    "asr_result": entry["asr_result"],
                    "author": author_name,
                    "work": entry["work"],
                    "rendition": rendition_name,
                    "line_num": entry["line_num"],
                    "analysis": entry["analysis"],
                    "reading_estimates": entry["reading_estimates"],
                })
            except Exception:
                continue

    inner.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--samples-per-parquet", type=int, default=50000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all (outer_zip_path, inner_name) tasks
    tasks = []
    for part_name in ["aozora_work_part1.zip", "aozora_work_part2.zip"]:
        part_path = input_dir / part_name
        if not part_path.exists():
            print(f"WARN: {part_path} not found, skipping")
            continue
        with zipfile.ZipFile(part_path) as outer:
            for inner_name in outer.namelist():
                if inner_name.endswith(".zip"):
                    tasks.append((str(part_path), inner_name))

    print(f"Total author zips to process: {len(tasks)}, workers: {args.num_workers}")

    # Process in parallel with bounded pool
    all_rows = []
    parquet_idx = 0
    total_samples = 0

    with Pool(args.num_workers) as pool:
        for rows in pool.imap_unordered(process_one_author, tasks):
            if not rows:
                continue
            all_rows.extend(rows)
            # Flush when buffer exceeds threshold
            while len(all_rows) >= args.samples_per_parquet:
                batch = all_rows[:args.samples_per_parquet]
                all_rows = all_rows[args.samples_per_parquet:]
                out_path = output_dir / f"train-{parquet_idx:05d}.parquet"
                pl.DataFrame(batch).write_parquet(out_path)
                total_samples += len(batch)
                print(f"Wrote {out_path.name} ({len(batch)} samples, total: {total_samples})")
                parquet_idx += 1

    # Flush remaining
    if all_rows:
        out_path = output_dir / f"train-{parquet_idx:05d}.parquet"
        pl.DataFrame(all_rows).write_parquet(out_path)
        total_samples += len(all_rows)
        print(f"Wrote {out_path.name} ({len(all_rows)} samples, total: {total_samples})")

    print(f"Done. {total_samples} total samples in {parquet_idx + 1} parquet files.")


if __name__ == "__main__":
    main()
