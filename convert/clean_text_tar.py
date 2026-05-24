import glob
import io
import os
import re
import tarfile


def clean_txt(content: bytes) -> bytes:
    text = content.decode("utf-8")
    # Strip user/assistant chat format if present
    marker = "assistant:"
    idx = text.find(marker)
    if idx != -1:
        text = text[idx + len(marker) :].lstrip(" ")
    # Remove image tokens like <|img1|>, <|img2|>, etc.
    text = re.sub(r"<\|img\d+\|>\s*", "", text)
    return text.encode("utf-8")


def process_dataset(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    tar_files = sorted(glob.glob(os.path.join(src_dir, "part-*.tar")))
    for src_path in tar_files:
        fname = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, fname)
        if os.path.exists(dst_path):
            print(f"Skipping {fname} (already done)")
            continue
        print(f"Processing {fname}...")
        try:
            with (
                tarfile.open(src_path, "r") as src_tar,
                tarfile.open(dst_path, "w") as dst_tar,
            ):
                for member in src_tar.getmembers():
                    f = src_tar.extractfile(member)
                    if f is None:
                        dst_tar.addfile(member)
                        continue
                    data = f.read()
                    if member.name.endswith(".txt"):
                        data = clean_txt(data)
                        info = tarfile.TarInfo(name=member.name)
                        info.size = len(data)
                        dst_tar.addfile(info, io.BytesIO(data))
                    else:
                        dst_tar.addfile(member, io.BytesIO(data))
        except tarfile.ReadError as e:
            print(f"WARNING: skipping {fname} — corrupted tar: {e}")
            if os.path.exists(dst_path):
                os.remove(dst_path)
    print("Done.")


import sys

if len(sys.argv) == 3:
    process_dataset(sys.argv[1], sys.argv[2])
else:
    print("Usage: clean.py <src_dir> <dst_dir>")
    sys.exit(1)
