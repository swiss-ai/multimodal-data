import logging
import os
import sqlite3

logger = logging.getLogger()


class Checkpoint:
    """Tracks last processed sample per dataset for resumability."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.conn = sqlite3.connect(db_path, timeout=10.0)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS progress (
                dataset_id TEXT PRIMARY KEY,
                last_sample_id TEXT,
                completed INTEGER DEFAULT 0
            )
        """)

    def get_resume_point(self, dataset_id: str) -> str | bool:
        """
        Returns the last processed sample ID for the dataset,
        True if completed, or False if not started.
        """

        logger.debug(f"Getting resume point for dataset_id={dataset_id}")

        cur = self.conn.execute(
            "SELECT completed, last_sample_id FROM progress WHERE dataset_id = ?",
            (dataset_id,),
        )

        row = cur.fetchone()
        if row is None:
            return False  # not started
        if row[0]:
            return True  # completed

        logger.info(f"Resuming from sample_id={row[0]} for dataset_id={dataset_id}")
        return row[1]

    def update(self, dataset_id: str, last_sample_id: str):
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO progress (dataset_id, last_sample_id, completed)
                VALUES (?, ?, 0)
                ON CONFLICT(dataset_id) DO UPDATE SET last_sample_id = ?
                """,
                (dataset_id, last_sample_id, last_sample_id),
            )

    def mark_complete(self, dataset_id: str):
        with self.conn:
            self.conn.execute(
                "UPDATE progress SET completed = 1 WHERE dataset_id = ?",
                (dataset_id,),
            )
