import os

import imagehash
import lmdb

from pipeline import BaseFilter, ImageSample, ImageTextSample, Sample


class HashStore:
    """LMDB store for image hashes."""

    def __init__(self, db_path: str, map_size: int = 1_000_000_000):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.env = lmdb.open(db_path, map_size=map_size)

    def check_and_insert(self, img_hash: str, dataset_id: str, sample_id: int) -> bool:
        """Insert hash if new. Returns True if unique, False if duplicate."""
        key = img_hash.encode()
        value = f"{dataset_id}:{sample_id}".encode()

        with self.env.begin(write=True) as tx:
            existing = tx.get(key)
            if existing is None:
                tx.put(key, value)
                return True  # unique hash
            return existing == value


class ImageDeduplication(BaseFilter):
    ALGORITHMS = {
        "phash": imagehash.phash,
        "dhash": imagehash.dhash,
        "ahash": imagehash.average_hash,
    }

    def __init__(self, db_path: str, algorithm: str):
        self.hash_store = HashStore(db_path)
        self.hash_func = self.ALGORITHMS[algorithm]

    def __call__(self, sample: Sample) -> bool:
        if not isinstance(sample, (ImageSample, ImageTextSample)):
            return True

        img_hash = str(self.hash_func(sample.image))
        is_unique = self.hash_store.check_and_insert(
            img_hash=img_hash,
            dataset_id=sample.meta.dataset_id,
            sample_id=sample.meta.sample_id,
        )

        return is_unique
