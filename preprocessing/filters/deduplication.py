import os

import imagehash
import lmdb

from pipeline import BaseFilter, ImageSample, ImageTextSample, Sample


class HashStore:
    """LMDB store for image hashes."""

    def __init__(self, db_path: str, map_size: int = 10_000_000_000):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.env = lmdb.open(db_path, map_size=map_size)

    def check_and_insert_batch(self, items: list[tuple[str, str, int]]) -> list[bool]:
        """
        Batch check/insert. Returns list of bools (True=unique or same sample, False=duplicate).
        items: a list of (img_hash, dataset_id, sample_id).
        """

        results = []
        with self.env.begin(write=True) as tx:
            for img_hash, dataset_id, sample_id in items:
                key = img_hash.encode()
                value = f"{dataset_id}:{sample_id}".encode()
                existing = tx.get(key)
                if existing is None:
                    tx.put(key, value)
                    results.append(True)
                else:
                    results.append(existing == value)
        return results


class ImageDeduplicationFilter(BaseFilter):
    ALGORITHMS = {
        "phash": imagehash.phash,
        "dhash": imagehash.dhash,
        "ahash": imagehash.average_hash,
    }

    def __init__(self, db_path: str, algorithm: str):
        self.hash_store = HashStore(db_path)
        self.hash_func = self.ALGORITHMS[algorithm]

    def process_batch(self, samples: list[Sample]) -> list[Sample]:
        """Batch dedup: local dedup first, then single LMDB transaction."""

        non_image_samples = []
        candidates = []  # (sample, hash)
        seen_hashes = set()

        for sample in samples:
            if not isinstance(sample, (ImageSample, ImageTextSample)):
                non_image_samples.append(sample)
                continue

            img_hash = str(self.hash_func(sample.image))
            if img_hash in seen_hashes:
                continue  # local duplicate

            seen_hashes.add(img_hash)
            candidates.append((sample, img_hash))

        # check against LMDB
        if candidates:
            items = [(h, s.meta.dataset_id, s.meta.sample_id) for s, h in candidates]
            db_results = self.hash_store.check_and_insert_batch(items)
            passed_images = [
                sample
                for (sample, _), is_unique in zip(candidates, db_results)
                if is_unique
            ]
        else:
            passed_images = []

        return non_image_samples + passed_images
