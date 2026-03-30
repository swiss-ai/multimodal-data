"""Extract audio-transcription pairs from NB Tale dataset.

Parses the XML annotation files and matches them to extracted WAV files.
Outputs a TSV manifest: audio_path, text, speaker, duration

Usage:
    python extract_transcriptions.py /capstor/.../nb-tale
"""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_xml(xml_path: str) -> list[dict]:
    """Parse an NB Tale XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    entries = []
    for ann in root.findall(".//annotation"):
        entry = {
            "id": ann.get("id"),
            "text": ann.get("text", ""),
            "end_time": float(ann.get("end", 0)),
            "speaker": ann.get("speaker", ""),
        }
        entries.append(entry)
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="NB Tale root directory")
    parser.add_argument("--output", default=None, help="Output TSV path (default: <root>/manifest.tsv)")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output) if args.output else root / "manifest.tsv"
    annotation_dir = root / "Annotation"

    # Collect all wav files indexed by their relative ID
    wav_index = {}
    for wav_path in root.glob("part_*/group_*/*.wav"):
        # e.g. part_1/group_01/p1_g01_f1_1_t-a0001.wav -> part_1/group_01/p1_g01_f1_1_t-a0001
        rel = wav_path.relative_to(root)
        key = str(rel.with_suffix(""))
        wav_index[key] = str(wav_path)

    print(f"Found {len(wav_index)} WAV files")

    # Parse all XML annotation files
    total = 0
    matched = 0
    rows = []

    for xml_file in sorted(annotation_dir.glob("part_*.xml")):
        entries = parse_xml(str(xml_file))
        print(f"  {xml_file.name}: {len(entries)} annotations")
        for entry in entries:
            total += 1
            aid = entry["id"]
            if aid in wav_index:
                matched += 1
                rows.append({
                    "audio_path": wav_index[aid],
                    "text": entry["text"],
                    "speaker": entry["speaker"],
                    "duration": entry["end_time"],
                })

    print(f"\nTotal annotations: {total}")
    print(f"Matched to WAV: {matched}")
    print(f"Unmatched: {total - matched}")

    # Write TSV
    with open(output, "w") as f:
        f.write("audio_path\ttext\tspeaker\tduration\n")
        for row in rows:
            text = row["text"].replace("\t", " ").replace("\n", " ")
            f.write(f"{row['audio_path']}\t{text}\t{row['speaker']}\t{row['duration']:.4f}\n")

    print(f"\nWrote {len(rows)} entries to {output}")


if __name__ == "__main__":
    main()
