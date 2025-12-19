import os
import sqlite3
from sqlite3 import Connection
from typing import List, Tuple


class AllowlistDB:
    """
    Manages the SQLite list of approved samples.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Connection

    def __enter__(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()
        return self

    def __exit__(self, *_):
        if self.conn:
            self.conn.commit()
            self.conn.close()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS allowlist (
                dataset_id TEXT,
                sample_id TEXT,
                PRIMARY KEY (dataset_id, sample_id)
            ) WITHOUT ROWID; 
        """)

    def add_batch(self, entries: List[Tuple[str, str]]):
        """
        Insert a batch of entries into the allowlist.

        Arguments:
            entries: list of (dataset_id, sample_id) pairs
        """
        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO allowlist (dataset_id, sample_id) VALUES (?, ?)",
                entries,
            )

    def add(self, dataset_id: str, sample_id: str):
        """
        Insert a single entry into the allowlist.
        """
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO allowlist (dataset_id, sample_id) VALUES (?, ?)",
                (dataset_id, sample_id),
            )

    def exists(self, dataset_id: str, sample_id: str) -> bool:
        """
        Check if sample (dataset_id, sample_id) is in the allowlist.
        """
        cur = self.conn.execute(
            "SELECT 1 FROM allowlist WHERE dataset_id = ? AND sample_id = ?",
            (dataset_id, sample_id),
        )
        return cur.fetchone() is not None

    def get_allowlist_for_dataset(self, dataset_id: str) -> List[str]:
        """
        Load the whitelisted IDs of the dataset into memory.
        Use with caution for large datasets.
        """
        cursor = self.conn.execute(
            "SELECT sample_id FROM allowlist WHERE dataset_id = ?",
            (dataset_id,),
        )
        return [row[0] for row in cursor]
