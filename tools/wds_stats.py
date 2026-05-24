import io
import os
import sys
from multiprocessing import Pool

import tiktoken
import webdataset as wds
from PIL import Image, PngImagePlugin

Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)  # 100 MB

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "tif", "tiff"}
TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(TOKEN_ENCODING.encode(text))


def normalize_sample_key(key):
    return key.rsplit(".", 1)[-1].lower()


def discover_tar_files(dataset):
    tar_files = []
    for root, _, files in os.walk(dataset):
        for filename in files:
            if filename.endswith(".tar") or filename.endswith(".tar.gz"):
                tar_files.append(os.path.join(root, filename))
    return sorted(tar_files)


def process_tar(tar_file):
    """Process a single tar file."""
    print(f"processing {tar_file}")
    ds = wds.WebDataset(tar_file, shardshuffle=False, handler=wds.warn_and_continue)
    total_width = 0
    total_height = 0
    total_images = 0
    total_tokens = 0
    num_samples = 0

    for sample in ds:
        num_samples += 1

        if "txt" in sample:
            text = sample["txt"].decode("utf-8")
            total_tokens += count_tokens(text)

        for key, value in sample.items():
            if key.startswith("__"):
                continue
            normalized_key = normalize_sample_key(key)
            if normalized_key in {"txt", "json"}:
                continue
            assert normalized_key in IMAGE_EXTENSIONS, f"unexpected key: {key}"
            img = Image.open(io.BytesIO(value))
            w, h = img.size
            total_width += w
            total_height += h
            total_images += 1

    return total_width, total_height, total_images, total_tokens, num_samples


def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    dataset = sys.argv[1]
    dataset_name = os.path.basename(dataset)

    tar_files = discover_tar_files(dataset)
    num_workers = min(os.cpu_count() or 1, len(tar_files), 200)
    with Pool(num_workers) as pool:
        results = pool.map(process_tar, tar_files)

    total_width = sum(r[0] for r in results)
    total_height = sum(r[1] for r in results)
    total_images = sum(r[2] for r in results)
    total_tokens = sum(r[3] for r in results)
    num_samples = sum(r[4] for r in results)

    avg_width = total_width / total_images if total_images > 0 else 0
    avg_height = total_height / total_images if total_images > 0 else 0
    avg_tokens_per_sample = total_tokens / num_samples if num_samples > 0 else 0
    avg_images_per_sample = total_images / num_samples if num_samples > 0 else 0

    with open(os.path.join(output_dir, f"{dataset_name}.txt"), "w") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"  Number of samples: {num_samples}\n")
        f.write(f"  Total images: {total_images}\n")
        f.write(f"  Avg images per sample: {avg_images_per_sample:.3f}\n")
        f.write(f"  Avg resolution: {avg_width:.3f} x {avg_height:.3f}\n")
        f.write(f"  Total tokens: {total_tokens}\n")
        f.write(f"  Avg tokens per sample: {avg_tokens_per_sample:.3f}\n")
        f.write(
            f"{dataset}: samples={num_samples} "
            f"imgs_per_sample={avg_images_per_sample:.2f} "
            f"avg_res={avg_width:.2f}x{avg_height:.2f} "
            f"tokens_per_sample={avg_tokens_per_sample:.1f}\n"
        )
        f.write(
            f"{dataset_name}\t{num_samples}\t{avg_tokens_per_sample:.2f}\t"
            f"{avg_images_per_sample:.2f}\t{avg_width:.2f}\t{avg_height:.2f}\n"
        )


if __name__ == "__main__":
    main()
