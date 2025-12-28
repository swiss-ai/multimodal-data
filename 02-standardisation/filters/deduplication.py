import os

import imagehash
import lmdb

from pipeline import BaseFilter, ImageSample, Sample


class HashStore:
    """LMDB store for image hashes."""

    def __init__(self, db_path: str, map_size: int = 10_000_000_000):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.env = lmdb.open(db_path, map_size=map_size)

    def check_and_insert_batch(self, items: list[tuple[str, str, int]]) -> list[bool]:
        """
        Batch check/insert. Returns list of bools (True=unique or same sample).
        Each item is (img_hash, dataset_id, sample_id).
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


class ImageDeduplication(BaseFilter):
    ALGORITHMS = {
        "phash": imagehash.phash,
        "dhash": imagehash.dhash,
        "ahash": imagehash.average_hash,
    }

    def __init__(self, db_path: str, algorithm: str):
        self.hash_store = HashStore(db_path)
        self.hash_func = self.ALGORITHMS[algorithm]

    def process_batch(self, samples: list[Sample]) -> list[bool]:
        """Batch dedup: local dedup first, then single LMDB transaction."""

        to_check = []  # (idx, hash, dataset_id, sample_id)

        # hash and local dedup
        seen_hashes = set()
        results = []
        for idx, sample in enumerate(samples):
            if not isinstance(sample, (ImageSample)):
                results.append(True)
                continue

            img_hash = str(self.hash_func(sample.image))
            if img_hash in seen_hashes:
                results.append(False)
                continue

            seen_hashes.add(img_hash)
            results.append(True)
            to_check.append(
                (
                    idx,
                    img_hash,
                    sample.meta.dataset_id,
                    sample.meta.sample_id,
                )
            )

        # check against LMDB
        if to_check:
            items = [(h, d, s) for _, h, d, s in to_check]
            db_results = self.hash_store.check_and_insert_batch(items)
            for (idx, _, _, _), is_unique in zip(to_check, db_results):
                assert results[idx]  # True from local dedup
                results[idx] = is_unique

        return results
