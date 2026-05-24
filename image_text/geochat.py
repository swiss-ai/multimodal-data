import bisect
import json
import os
import zipfile

import webdataset as wds

SNAPSHOT = os.getenv("GEOCHAT_SNAPSHOT", "/path/to/data")
JSON_PATH = os.path.join(SNAPSHOT, "GeoChat_Instruct.json")
ZIP_PARTS = [
    os.path.join(SNAPSHOT, "images_partaa"),
    os.path.join(SNAPSHOT, "images_partab"),
    os.path.join(SNAPSHOT, "images_partac"),
]
ZIP_PREFIX = os.getenv("GEOCHAT_ZIP_PREFIX", "")
OUT_DIR = os.getenv("GEOCHAT_OUT_DIR", "/path/to/output")

START_INDEX = int(os.getenv("GEOCHAT_START_INDEX", "0"))
START_SHARD = int(os.getenv("GEOCHAT_START_SHARD", "0"))


class MultiFile:
    """Presents multiple files as a single seekable file-like object."""

    def __init__(self, paths):
        self._fds = [open(p, "rb") for p in paths]
        sizes = [os.path.getsize(p) for p in paths]
        self.total_size = sum(sizes)
        self._cum = [0]
        for s in sizes:
            self._cum.append(self._cum[-1] + s)
        self._pos = 0

    def _file_and_offset(self, pos):
        i = bisect.bisect_right(self._cum, pos) - 1
        i = max(0, min(i, len(self._fds) - 1))
        return i, pos - self._cum[i]

    def read(self, n=-1):
        if n == -1:
            n = self.total_size - self._pos
        chunks = []
        while n > 0 and self._pos < self.total_size:
            i, off = self._file_and_offset(self._pos)
            self._fds[i].seek(off)
            to_read = min(n, self._cum[i + 1] - self._pos)
            chunk = self._fds[i].read(to_read)
            if not chunk:
                break
            chunks.append(chunk)
            self._pos += len(chunk)
            n -= len(chunk)
        return b"".join(chunks)

    def seek(self, pos, whence=0):
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = self.total_size + pos
        self._pos = max(0, min(self._pos, self.total_size))
        return self._pos

    def tell(self):
        return self._pos

    def seekable(self):
        return True

    def close(self):
        for fd in self._fds:
            fd.close()


def embed_placeholders(text, n_images):
    placeholders = [f"<|img{i}|>" for i in range(1, n_images + 1)]
    missing = [p for p in placeholders if p not in text]
    if missing:
        text = "\n".join(missing) + "\n" + text
    return text


def format_conversation(conversations):
    parts = []
    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        value = turn["value"].replace("<image>", "<|img1|>").strip()
        parts.append(f"{role}: {value}")
    return "\n".join(parts)


def get_ext(image_path):
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext in ("jpeg", "jpg"):
        return "jpg"
    return ext


def main():
    print(f"Loading {JSON_PATH}...")
    with open(JSON_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items.")

    print("Opening zip archive...")
    mf = MultiFile(ZIP_PARTS)
    zf = zipfile.ZipFile(mf)

    # Build lookup: relative image path -> zip entry name
    zip_lookup = {}
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        rel = name[len(ZIP_PREFIX) :] if name.startswith(ZIP_PREFIX) else name
        zip_lookup[rel] = name
    print(f"  ZIP contains {len(zip_lookup)} files.")

    os.makedirs(OUT_DIR, exist_ok=True)
    pattern = os.path.join(OUT_DIR, "part-%06d.tar")
    sink = wds.ShardWriter(pattern, maxcount=10000, start_shard=START_SHARD)

    total_written = 0
    total_missing = 0

    for i, item in enumerate(data[START_INDEX:], start=START_INDEX):
        zip_name = zip_lookup.get(item["image"])
        if zip_name is None:
            total_missing += 1
            if total_missing <= 10:
                print(f"  WARNING: image not found in zip: {item['image']}")
            continue

        image_bytes = zf.read(zip_name)
        ext = get_ext(item["image"])
        text = format_conversation(item["conversations"])
        text = embed_placeholders(text, n_images=1)
        key = f"geochat__{i:06d}"

        sink.write(
            {
                "__key__": key,
                f"img1.{ext}": image_bytes,
                "txt": text.encode("utf-8"),
            }
        )
        total_written += 1

        if (i + 1) % 50000 == 0:
            print(f"  Progress: {i + 1}/{len(data)}, written={total_written}, missing={total_missing}")

    sink.close()
    zf.close()
    mf.close()
    print(f"\nFinished. Written: {total_written}, missing images: {total_missing}")


if __name__ == "__main__":
    main()
