import sqlite3
from sqlite3 import Connection
from typing import List, Tuple


class AllowlistDB:
    """
    Manages the SQLite list of approved samples.
    Schema:
        - dataset_name (TEXT)
        - sample_id (TEXT)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Connection

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.create_table()

        # allow fast writes, data loss is acceptable on crash
        self.conn.execute("PRAGMA synchronous = OFF")
        # allow concurrent reads/writes
        self.conn.execute("PRAGMA journal_mode = WAL")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.commit()
            self.conn.close()

    def create_table(self):
        # only store keys; if a key is here, it's approved
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS allowlist (
                dataset_name TEXT,
                sample_id TEXT,
                PRIMARY KEY (dataset_name, sample_id)
            ) WITHOUT ROWID; 
        """)

    def add_batch(self, entries: List[Tuple[str, str]]):
        """
        Insert a batch of entries into the allowlist.

        entries: list of (dataset_name, sample_id)
        """
        with self.conn:
            self.conn.executemany(
                "INSERT INTO allowlist (dataset_name, sample_id) VALUES (?, ?)",
                entries,
            )

    def add(self, dataset_name: str, sample_id: str):
        """
        Insert a single entry into the allowlist.
        """
        with self.conn:
            self.conn.execute(
                "INSERT INTO allowlist (dataset_name, sample_id) VALUES (?, ?)",
                (dataset_name, sample_id),
            )

    def exists(self, dataset_name: str, sample_id: str) -> bool:
        """
        Check if (dataset_name, sample_id) is in the allowlist.
        """
        cur = self.conn.execute(
            "SELECT 1 FROM allowlist WHERE dataset_name = ? AND sample_id = ?",
            (dataset_name, sample_id),
        )
        return cur.fetchone() is not None

    def get_allowlist_for_dataset(self, dataset_name: str) -> List[str]:
        """
        Load the entire allowed ID set for a dataset into memory.
        Use with caution for large datasets.
        """
        cursor = self.conn.execute(
            "SELECT sample_id FROM allowlist WHERE dataset_name = ?",
            (dataset_name,),
        )
        return [row[0] for row in cursor]
