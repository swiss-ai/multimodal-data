"""Filter audiocite by license and split into separate directories.

1. Deletes directories with NC or ND licenses.
2. Splits remaining into wavs_cc_by/ and wavs_cc_by_sa/.

Usage:
    python filter_license.py /capstor/.../audiocite
    python filter_license.py /capstor/.../audiocite --dry-run
"""

import argparse
import csv
import shutil
from pathlib import Path


def classify_directories(csv_path):
    """Classify directories by license type."""
    cc_by = []
    cc_by_sa = []
    to_delete = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # skip header
        for row in reader:
            license_str = row[4]
            directory = row[7].strip()
            has_nc = "NC" in license_str or "commerciale" in license_str.lower()
            has_nd = "ND" in license_str or "modification" in license_str.lower()
            if has_nc or has_nd:
                to_delete.append(directory)
            elif "SA" in license_str or "Partage" in license_str:
                cc_by_sa.append(directory)
            else:
                cc_by.append(directory)

    return cc_by, cc_by_sa, to_delete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Audiocite root directory")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without doing it")
    args = parser.parse_args()

    root = Path(args.root)
    csv_path = root / "metadata" / "downloaded.csv"

    cc_by, cc_by_sa, to_delete = classify_directories(csv_path)

    print(f"CC-BY:    {len(cc_by)} directories")
    print(f"CC-BY-SA: {len(cc_by_sa)} directories")
    print(f"NC/ND:    {len(to_delete)} directories (to delete)")

    # Step 1: Delete NC/ND directories
    deleted_files = 0
    deleted_bytes = 0
    for d in to_delete:
        full = root / d
        if full.is_dir():
            for fp in full.rglob("*"):
                if fp.is_file():
                    deleted_files += 1
                    deleted_bytes += fp.stat().st_size
            if not args.dry_run:
                shutil.rmtree(full)

    action = "Would delete" if args.dry_run else "Deleted"
    print(f"\n{action}: {deleted_files} files ({deleted_bytes / (1024**3):.1f} GB)")

    # Step 2: Split remaining into wavs_cc_by/ and wavs_cc_by_sa/
    if not args.dry_run:
        (root / "wavs_cc_by").mkdir(exist_ok=True)
        (root / "wavs_cc_by_sa").mkdir(exist_ok=True)

    moved = 0
    missing = 0
    for directory, target in [(d, "wavs_cc_by") for d in cc_by] + [(d, "wavs_cc_by_sa") for d in cc_by_sa]:
        src = root / directory
        if src.is_dir():
            subdir = Path(directory).relative_to("wavs")
            dest = root / target / subdir
            if args.dry_run:
                print(f"  [dry-run] {src} -> {dest}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
            moved += 1
        else:
            missing += 1

    action = "Would move" if args.dry_run else "Moved"
    print(f"\n{action}: {moved} directories")
    if missing:
        print(f"Missing: {missing} (already deleted or moved)")

    # Remove empty wavs/ dir
    if not args.dry_run:
        wavs_dir = root / "wavs"
        if wavs_dir.exists() and not any(wavs_dir.iterdir()):
            wavs_dir.rmdir()
            print("Removed empty wavs/ directory")


if __name__ == "__main__":
    main()
