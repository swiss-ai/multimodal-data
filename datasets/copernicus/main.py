import io
import os
import subprocess
import tarfile
import zipfile

import numpy as np
import pandas as pd
import rasterio
import webdataset as wds
from PIL import Image

SNAP = "/path/to/data/vision-datasets/hf_hub_cache/datasets--wangyi111--Copernicus-Bench/snapshots/a287ab1b414d2bff99557166988571c5885ed81a"
BLOB_DIR = "/path/to/data/vision-datasets/hf_hub_cache/datasets--wangyi111--Copernicus-Bench/blobs"
OUTPUT_DIR = "/tmp/copernicus"
SAMPLES_PER_SHARD = 10000
CLIP_MAX = 2500


def make_rgb_png_bytes(tif_data):
    """Convert 13-band S2 tif bytes to RGB PNG bytes."""
    with rasterio.open(io.BytesIO(tif_data)) as src:
        r = src.read(4).astype(np.float32)
        g = src.read(3).astype(np.float32)
        b = src.read(2).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        r[r == nodata] = np.nan
        g[g == nodata] = np.nan
        b[b == nodata] = np.nan

    rgb = np.dstack((r, g, b))
    rgb = np.clip(rgb, 0, CLIP_MAX)
    rgb = (rgb / CLIP_MAX * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def make_rgb_png_from_bands(b04_data, b03_data, b02_data):
    """Convert separate B04/B03/B02 band tif bytes to RGB PNG bytes (upscaled to 240x240)."""
    bands = []
    for data in (b04_data, b03_data, b02_data):
        with rasterio.open(io.BytesIO(data)) as src:
            band = src.read(1).astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            band[band == nodata] = np.nan
        bands.append(band)

    rgb = np.dstack(bands)
    rgb = np.clip(rgb, 0, CLIP_MAX)
    rgb = (rgb / CLIP_MAX * 255).astype(np.uint8)

    img = Image.fromarray(rgb).resize((240, 240), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def write_to_wds(dataset_name, sample_iter):
    """Write samples to webdataset shards.

    sample_iter yields (key, png_bytes) tuples.
    """
    out_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, f"{dataset_name}-%06d.tar")

    sink = wds.ShardWriter(pattern, maxcount=SAMPLES_PER_SHARD)
    count = 0

    for key, png_bytes in sample_iter:
        sink.write({"__key__": key, "png": png_bytes})
        count += 1
        if count % 500 == 0:
            print(f"  [{dataset_name}] {count} samples written...")

    sink.close()
    print(f"  [{dataset_name}] Done: {count} samples total")


def iter_bigearthnet_full():
    """Yield (key, png_bytes) from full BigEarthNet-S2.tar.zst with individual band tifs.

    Reads B04, B03, B02 per patch (120x120 at 10m) and combines to RGB.
    Filters to the 5% metadata subset (24,002 patches).
    """
    # Load 5% metadata for filtering
    meta_path = os.path.join(SNAP, "l2_bigearthnet_s1s2/metadata-5%.parquet")
    valid_ids = set(pd.read_parquet(meta_path)["patch_id"].values)
    print(f"  Filtering to {len(valid_ids)} patches from 5% metadata")

    blob = "/path/to/data/vision-datasets/BigEarthNet/BigEarthNet-S2.tar.zst"
    proc = subprocess.Popen(
        ["zstd", "-d", "-c", blob],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    # Bands are grouped by patch directory, appearing in order B01..B12,B8A
    # We only need B02, B03, B04
    RGB_BANDS = {"B02", "B03", "B04"}
    current_patch = None
    band_data = {}
    matched = 0
    skipped = 0

    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        for member in tar:
            if not member.name.endswith(".tif"):
                continue

            basename = os.path.basename(member.name)  # ..._26_57_B04.tif
            band = basename.rsplit("_", 1)[1].replace(".tif", "")  # B04
            # patch_id: everything before _B0x
            patch_id = basename.rsplit("_", 1)[0]  # S2A_..._26_57

            # New patch?
            if patch_id != current_patch:
                # Emit previous if complete
                if current_patch and len(band_data) == 3:
                    png = make_rgb_png_from_bands(band_data["B04"], band_data["B03"], band_data["B02"])
                    matched += 1
                    yield current_patch, png
                current_patch = patch_id
                band_data = {}

            if band in RGB_BANDS and patch_id in valid_ids:
                f = tar.extractfile(member)
                if f is not None:
                    band_data[band] = f.read()
            else:
                # Skip non-RGB bands and non-matching patches
                skipped += 1

    # Don't forget the last patch
    if current_patch and len(band_data) == 3:
        png = make_rgb_png_from_bands(band_data["B04"], band_data["B03"], band_data["B02"])
        matched += 1
        yield current_patch, png

    proc.terminate()
    proc.wait()
    print(f"  Matched {matched} patches, skipped {skipped} band files")


def iter_zip_as_png(zip_path, tif_filter=None):
    """Yield (key, png_bytes) from a zip file."""
    with zipfile.ZipFile(zip_path) as z:
        tifs = [n for n in z.namelist() if n.endswith(".tif")]
        if tif_filter:
            tifs = [n for n in tifs if tif_filter(n)]
        for name in tifs:
            key = os.path.splitext(os.path.basename(name))[0]
            yield key, make_rgb_png_bytes(z.read(name))


if __name__ == "__main__":
    # 1) EuroSAT S2 (zip) — 27,000 images, 64x64
    print("=== EuroSAT S2 ===")
    write_to_wds(
        "eurosat",
        iter_zip_as_png(os.path.join(SNAP, "l2_eurosat_s1s2/eurosat_s2.zip")),
    )

    # 2) DFC2020 S2 only (zip) — 5,128 images, 256x256
    print("\n=== DFC2020 S2 ===")
    write_to_wds(
        "dfc2020",
        iter_zip_as_png(
            os.path.join(SNAP, "l2_dfc2020_s1s2/dfc2020.zip"),
            tif_filter=lambda n: "/s2/" in n,
        ),
    )

    # 3) BigEarthNet S2 (full archive) — 24,002 images, 120x120 native
    print("\n=== BigEarthNet S2 ===")
    write_to_wds("bigearthnet", iter_bigearthnet_full())
