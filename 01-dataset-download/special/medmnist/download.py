"""
Download MedMNIST 224-px variants from Zenodo (record 10519652).

Usage:
    pip install zenodo_get
    python download.py --output-dir /path/to/medmnist
"""

import argparse

from zenodo_get import download


def main():
    parser = argparse.ArgumentParser(description="Download MedMNIST from Zenodo")
    parser.add_argument(
        "--output-dir",
        default="./medmnist",
        help="Directory to save the downloaded files",
    )
    args = parser.parse_args()

    download(
        record_or_doi="10519652",
        output_dir=args.output_dir,
        file_glob="*_224.npz",
    )
    print(f"MedMNIST downloaded to {args.output_dir}")


if __name__ == "__main__":
    main()
