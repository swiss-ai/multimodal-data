import logging
import os
import sqlite3

logger = logging.getLogger()


class Allowlist:
    """Manages the manifest of approved sample IDs."""

    def __init__(self, db_path: str):
        self.db_path = db_path

        logger.info(f"Initializing allowlist database at {db_path}")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, timeout=10.0)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._create_table()

    def _create_table(self):
        logger.debug("Creating allowlist table if not exists")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS allowlist (
                dataset_id TEXT,
                sample_id TEXT,
                PRIMARY KEY (dataset_id, sample_id)
            ) WITHOUT ROWID
        """)

    def add_batch(self, entries: list[tuple[str, str]]):
        """Insert batch of (dataset_id, sample_id) pairs."""
        logger.debug(f"Inserting {len(entries)} entries into allowlist")

        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO allowlist (dataset_id, sample_id) VALUES (?, ?)",
                entries,
            )

    def exists(self, dataset_id: str, sample_id: str) -> bool:
        """Check if sample is in allowlist."""
        cur = self.conn.execute(
            "SELECT 1 FROM allowlist WHERE dataset_id = ? AND sample_id = ?",
            (dataset_id, sample_id),
        )
        return cur.fetchone() is not None

    def iter_dataset(self, dataset_id: str):
        """Yields dataset sample IDs in the allowlist."""
        logger.debug(f"Iterating allowlist for dataset_id={dataset_id}")

        cursor = self.conn.execute(
            "SELECT sample_id FROM allowlist WHERE dataset_id = ?",
            (dataset_id,),
        )
        for row in cursor:
            yield row[0]
