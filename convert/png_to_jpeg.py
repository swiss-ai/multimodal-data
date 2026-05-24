#!/usr/bin/env python3

import argparse
import io
import sys
from pathlib import Path

import webdataset as wds
from PIL import Image, ImageOps

TARGET_SIZE = (512, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a WebDataset directory of tar shards and convert all .png payloads "
            "to resized high-quality .jpg payloads while preserving keys and non-image fields."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing input tar shards")
    parser.add_argument("output_dir", type=Path, help="Directory where converted tar shards are written")
    parser.add_argument(
        "--pattern",
        default="part-*.tar",
        help="Glob pattern used to select shards inside input_dir (default: %(default)s)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=100,
        help="JPEG quality passed to Pillow (default: %(default)s)",
    )
    parser.add_argument(
        "--subsampling",
        type=int,
        default=0,
        help="JPEG chroma subsampling passed to Pillow (default: %(default)s)",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Only process the first N shards, useful for smoke tests",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after writing N samples total, useful for smoke tests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output shards if they already exist",
    )
    return parser.parse_args()


def discover_shards(input_dir: Path, pattern: str) -> list[Path]:
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    shards = sorted(input_dir.glob(pattern))
    if not shards:
        raise SystemExit(f"No shards matched {pattern!r} in {input_dir}")
    return shards


def image_to_rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGB", "L"}:
        return image.convert("RGB")
    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def resize_image(image: Image.Image) -> Image.Image:
    return image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)


def png_bytes_to_jpeg(raw: bytes, quality: int, subsampling: int) -> bytes:
    with Image.open(io.BytesIO(raw)) as image:
        rgb = resize_image(image_to_rgb(image))
        output = io.BytesIO()
        rgb.save(
            output,
            format="JPEG",
            quality=quality,
            subsampling=subsampling,
            optimize=True,
        )
        return output.getvalue()


def normalize_payload(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError(f"Unsupported payload type: {type(value)!r}")


def process_shard(
    input_shard: Path,
    output_shard: Path,
    quality: int,
    subsampling: int,
    sample_budget: int | None,
) -> tuple[int, int, int, bool]:
    written_samples = 0
    bytes_in = 0
    bytes_out = 0
    stopped_early = False

    dataset = wds.WebDataset(str(input_shard), shardshuffle=False)
    with wds.TarWriter(str(output_shard), encoder=False) as sink:
        for sample in dataset:
            output_sample = {"__key__": sample["__key__"]}
            for key, value in sample.items():
                if key.startswith("__"):
                    continue

                payload = normalize_payload(value)
                bytes_in += len(payload)

                if key.endswith(".png"):
                    new_key = f"{key[:-4]}.jpg"
                    converted = png_bytes_to_jpeg(payload, quality=quality, subsampling=subsampling)
                    output_sample[new_key] = converted
                    bytes_out += len(converted)
                else:
                    output_sample[key] = payload
                    bytes_out += len(payload)

            sink.write(output_sample)
            written_samples += 1

            if sample_budget is not None and written_samples >= sample_budget:
                stopped_early = True
                break

    return written_samples, bytes_in, bytes_out, stopped_early


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if input_dir == output_dir:
        raise SystemExit("Input and output directories must be different")

    shards = discover_shards(input_dir, args.pattern)
    if args.max_shards is not None:
        shards = shards[: args.max_shards]

    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    total_in = 0
    total_out = 0

    for shard_index, input_shard in enumerate(shards, start=1):
        output_shard = output_dir / input_shard.name
        if output_shard.exists() and not args.overwrite:
            print(f"[skip] {output_shard} already exists", file=sys.stderr)
            continue

        if output_shard.exists() and args.overwrite:
            output_shard.unlink()

        remaining_budget = None
        if args.max_samples is not None:
            remaining_budget = args.max_samples - total_samples
            if remaining_budget <= 0:
                break

        print(
            f"[{shard_index}/{len(shards)}] {input_shard.name} -> {output_shard.name}",
            file=sys.stderr,
        )
        written, bytes_in, bytes_out, stopped_early = process_shard(
            input_shard=input_shard,
            output_shard=output_shard,
            quality=args.quality,
            subsampling=args.subsampling,
            sample_budget=remaining_budget,
        )
        total_samples += written
        total_in += bytes_in
        total_out += bytes_out

        ratio = (bytes_out / bytes_in) if bytes_in else 0.0
        print(
            f"  samples={written} input_bytes={bytes_in} output_bytes={bytes_out} ratio={ratio:.4f}",
            file=sys.stderr,
        )

        if stopped_early:
            break

    overall_ratio = (total_out / total_in) if total_in else 0.0
    savings_ratio = 1.0 - overall_ratio if total_in else 0.0
    print(
        (
            f"done samples={total_samples} input_bytes={total_in} output_bytes={total_out} "
            f"ratio={overall_ratio:.4f} savings={savings_ratio:.4%}"
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
