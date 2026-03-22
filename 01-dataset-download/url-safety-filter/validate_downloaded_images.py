"""Post-download image validation to catch malicious content that bypassed URL filtering.

Checks:
  1. Magic bytes — verify file is actually an image (not an executable/script)
  2. Image decodability — file must open as a valid image via PIL
  3. Dimension sanity — reject degenerate images (0px, or > 20000px)
  4. File-size sanity — reject suspiciously tiny (<100B) or huge (>200MB) files
  5. Extension mismatch — file extension vs actual format from magic bytes

Usage:
    python validate_downloaded_images.py \
        --input-dir /path/to/downloaded_images/ \
        --workers 64 \
        --quarantine-dir /path/to/quarantine/
"""

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Magic bytes for common image formats
IMAGE_MAGIC = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"RIFF": "webp",  # RIFF....WEBP
    b"BM": "bmp",
    b"II\x2a\x00": "tiff",
    b"MM\x00\x2a": "tiff",
}

FORMAT_EXTENSIONS = {
    "jpeg": {".jpg", ".jpeg"},
    "png": {".png"},
    "gif": {".gif"},
    "webp": {".webp"},
    "bmp": {".bmp"},
    "tiff": {".tif", ".tiff"},
}

DEFAULT_EXPECTED_EXTENSIONS = ".jpg,.jpeg,.png,.gif,.webp,.bmp,.tiff,.tif"
DEFAULT_IGNORED_EXTENSIONS = ".json,.jsonl,.txt,.csv,.tsv,.parquet,.arrow,.md,.yaml,.yml"

MIN_FILE_SIZE = 100  # bytes
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB
MAX_DIMENSION = 20000  # pixels
MAX_IMAGE_PIXELS = 100_000_000  # 100 megapixels


def check_magic_bytes(data: bytes) -> str | None:
    """Return detected image format from magic bytes, or None if not an image."""
    for magic, fmt in IMAGE_MAGIC.items():
        if data.startswith(magic):
            if fmt == "webp" and len(data) >= 12:
                if data[8:12] != b"WEBP":
                    return None
            return fmt
    return None


def discover_input_files(
    input_dir: Path,
    ignored_extensions: set[str],
    expected_extensions: set[str],
    quarantine_dir: Path | None = None,
    extensions_only: bool = False,
) -> list[Path]:
    """Find files to validate, scanning broadly by default to catch mislabeled payloads."""
    files = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if quarantine_dir is not None and path.is_relative_to(quarantine_dir):
            continue
        if path.suffix.lower() in ignored_extensions:
            continue
        if extensions_only and path.suffix.lower() not in expected_extensions:
            continue
        files.append(path)
    return files


def extension_matches_format(path: Path, fmt: str) -> bool:
    """Return whether a file suffix matches its detected image format."""
    suffix = path.suffix.lower()
    return suffix in FORMAT_EXTENSIONS.get(fmt, set())


def validate_image(path: Path) -> tuple[str, str | None]:
    """Validate a single image file. Returns (path, failure_reason) or (path, None) if OK."""
    try:
        size = path.stat().st_size
        if size < MIN_FILE_SIZE:
            return str(path), f"too_small ({size}B)"
        if size > MAX_FILE_SIZE:
            return str(path), f"too_large ({size / 1024 / 1024:.1f}MB)"

        with open(path, "rb") as f:
            header = f.read(32)

        fmt = check_magic_bytes(header)
        if fmt is None:
            return str(path), f"bad_magic (first bytes: {header[:8].hex()})"
        if not path.suffix:
            return str(path), f"missing_extension ({fmt})"
        if not extension_matches_format(path, fmt):
            return str(path), f"extension_mismatch ({path.suffix.lower()} != {fmt})"

        # Try to actually decode the image
        from PIL import Image
        previous_max_pixels = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as img:
                    w, h = img.size
                    if w == 0 or h == 0:
                        return str(path), f"zero_dimension ({w}x{h})"
                    if w > MAX_DIMENSION or h > MAX_DIMENSION:
                        return str(path), f"huge_dimension ({w}x{h})"
                    if w * h > MAX_IMAGE_PIXELS:
                        return str(path), f"too_many_pixels ({w}x{h})"
                    # Force full decode to catch truncated/corrupt files
                    img.load()
        finally:
            Image.MAX_IMAGE_PIXELS = previous_max_pixels

        return str(path), None

    except Exception as e:
        if type(e).__name__ in {"DecompressionBombWarning", "DecompressionBombError"}:
            return str(path), f"decompression_bomb ({e})"
        return str(path), f"decode_error ({type(e).__name__}: {e})"


def main():
    global MAX_IMAGE_PIXELS

    parser = argparse.ArgumentParser(description="Validate downloaded images")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--quarantine-dir", type=Path, help="Move bad files here instead of deleting")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't move/delete")
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=MAX_IMAGE_PIXELS,
        help="Maximum allowed pixel count before treating an image as suspicious",
    )
    parser.add_argument(
        "--extensions",
        default=DEFAULT_EXPECTED_EXTENSIONS,
        help="Comma-separated image extensions considered valid for extension matching",
    )
    parser.add_argument(
        "--ignore-extensions",
        default=DEFAULT_IGNORED_EXTENSIONS,
        help="Comma-separated metadata extensions to skip while scanning",
    )
    parser.add_argument(
        "--extensions-only",
        action="store_true",
        help="Only validate files whose suffix is listed in --extensions",
    )
    args = parser.parse_args()
    MAX_IMAGE_PIXELS = args.max_pixels

    extensions = set(args.extensions.lower().split(","))
    ignored_extensions = set(args.ignore_extensions.lower().split(","))
    image_files = discover_input_files(
        args.input_dir,
        ignored_extensions=ignored_extensions,
        expected_extensions=extensions,
        quarantine_dir=args.quarantine_dir,
        extensions_only=args.extensions_only,
    )
    print(f"Found {len(image_files):,} image files in {args.input_dir}")

    bad_files = []
    checked = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(validate_image, p): p for p in image_files}
        for future in as_completed(futures):
            path, reason = future.result()
            checked += 1
            if reason:
                bad_files.append((path, reason))
            if checked % 50000 == 0 or checked == len(image_files):
                print(f"  [{checked:,}/{len(image_files):,}] bad so far: {len(bad_files):,}")

    print(f"\nResults: {checked:,} checked, {len(bad_files):,} bad ({len(bad_files) / max(checked, 1) * 100:.2f}%)")

    if bad_files:
        # Group by reason
        reasons = {}
        for path, reason in bad_files:
            key = reason.split(" ")[0]
            reasons.setdefault(key, []).append(path)
        print("\nBreakdown:")
        for reason, paths in sorted(reasons.items(), key=lambda x: -len(x[1])):
            print(f"  {reason}: {len(paths):,}")

        if not args.dry_run and args.quarantine_dir:
            args.quarantine_dir.mkdir(parents=True, exist_ok=True)
            for path, reason in bad_files:
                src = Path(path)
                dst = args.quarantine_dir / src.relative_to(args.input_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
            print(f"\nMoved {len(bad_files):,} files to {args.quarantine_dir}")
        elif not args.dry_run:
            for path, _ in bad_files:
                Path(path).unlink()
            print(f"\nDeleted {len(bad_files):,} files")
        else:
            # Write report
            report = args.input_dir / "validation_report.txt"
            with open(report, "w") as f:
                for path, reason in sorted(bad_files):
                    f.write(f"{reason}\t{path}\n")
            print(f"\nDry run — report written to {report}")

    print("Done.")


if __name__ == "__main__":
    main()
