import os

from zenodo_get import download

download(
    record_or_doi="10519652",
    output_dir=os.environ.get("OUTPUT_DIR", "/path/to/data/medical/raw/medmnist"),
    file_glob="*_224.npz",
)
