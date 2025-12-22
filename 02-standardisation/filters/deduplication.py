import os
import sqlite3

import imagehash

from src.base import BaseFilter
from src.schema import ImageSample, ImageTextSample, Sample


class HashStore:
    """SQLite store for image hashes."""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)

        try:
            self.conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.OperationalError:
            pass  # WAL already called or not supported

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS seen_hashes (
                img_hash TEXT PRIMARY KEY,
                dataset_id TEXT,
                sample_id TEXT
            ) WITHOUT ROWID
        """)

    def check_and_insert(self, img_hash: str, dataset_id: str, sample_id: str) -> bool:
        """Insert hash if new. Returns True if unique, False if duplicate."""
        with self.conn:
            cursor = self.conn.execute(
                "INSERT OR IGNORE INTO seen_hashes (img_hash, dataset_id, sample_id) VALUES (?, ?, ?)",
                (img_hash, dataset_id, sample_id),
            )
            return cursor.rowcount > 0


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
