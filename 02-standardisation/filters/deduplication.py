import os
import sqlite3

import imagehash
from PIL import Image

from src.base import BaseFilter
from src.schema import ImageSample, RawSample


class HashStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")  # better concurrency
        self.conn.execute("PRAGMA synchronous=NORMAL;")  # performance
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS seen_hashes (
                    img_hash TEXT PRIMARY KEY,
                    dataset_id TEXT,
                    sample_id TEXT
                ) WITHOUT ROWID;
            """)

    def check_and_insert(self, img_hash: str, dataset_id: str, sample_id: str) -> bool:
        """
        Check if the hash exists; if not, insert it atomically.

        Args:
            img_hash (str): The image hash to check and insert.
            dataset_id (str): The dataset identifier.
            sample_id (str): The sample identifier.

        Returns:
            bool: True if the hash was inserted (not seen before)
                  False if it was already present.
        """
        try:
            with self.conn:
                cursor = self.conn.execute(
                    """
                    INSERT OR IGNORE INTO seen_hashes 
                    (img_hash, dataset_id, sample_id) 
                    VALUES (?, ?, ?)
                    """,
                    (img_hash, dataset_id, sample_id),
                )
                return cursor.rowcount > 0
        except sqlite3.OperationalError as e:
            print(f"SQLite OperationalError: {e}")
            return False


class ImageDeduplication(BaseFilter):
    def __init__(
        self,
        db_path: str,
        algorithm: str = "phash",
    ):
        self.hash_store = HashStore(db_path)

        self.algorithm = algorithm
        self.algorithm_funcs = {
            "phash": imagehash.phash,
            "dhash": imagehash.dhash,
            "ahash": imagehash.average_hash,
        }
        assert self.algorithm in self.algorithm_funcs

    def compute_hash(self, img: Image.Image) -> str:
        hash_func = self.algorithm_funcs[self.algorithm]
        img_hash = hash_func(img)
        return str(img_hash)

    def __call__(self, sample: RawSample) -> bool:
        if not isinstance(sample, ImageSample):
            return True

        img_hash = self.compute_hash(sample.image)
        is_unique = self.hash_store.check_and_insert(
            img_hash=img_hash,
            dataset_id=sample.meta.dataset_id,
            sample_id=sample.meta.sample_id,
        )

        return is_unique
