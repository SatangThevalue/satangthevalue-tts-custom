import os
from pathlib import Path
import sqlite3
import time
from typing import Dict, Any, List, Optional

from src.utils.logger import setup_logger

logger = setup_logger("media_registry")

DEFAULT_DB_PATH = "/content/drive/MyDrive/tts-project/registry.db"


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates directory and establishes SQLite connection with WAL mode."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_registry_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes schema for tracking media ingestion, duration, and pipeline status."""
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS media_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_identifier TEXT NOT NULL,
                raw_path TEXT NOT NULL UNIQUE,
                duration_sec REAL DEFAULT 0.0,
                status TEXT NOT NULL DEFAULT 'raw',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_speaker ON media_registry(speaker_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON media_registry(source_identifier);")
    logger.debug(f"Registry SQLite database initialized at: {db_path}")


def is_already_registered(
    source_identifier: str,
    speaker_id: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Checks whether a media URL or file was already ingested."""
    init_registry_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if speaker_id:
            cursor.execute(
                "SELECT id FROM media_registry WHERE source_identifier = ? AND speaker_id = ? LIMIT 1;",
                (source_identifier, speaker_id),
            )
        else:
            cursor.execute(
                "SELECT id FROM media_registry WHERE source_identifier = ? LIMIT 1;",
                (source_identifier,),
            )
        row = cursor.fetchone()
        return row is not None


def record_media_item(
    speaker_id: str,
    source_type: str,
    source_identifier: str,
    raw_path: str,
    duration_sec: float = 0.0,
    status: str = "raw",
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Inserts or updates an ingested media record."""
    init_registry_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO media_registry (speaker_id, source_type, source_identifier, raw_path, duration_sec, status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(raw_path) DO UPDATE SET
                speaker_id = excluded.speaker_id,
                duration_sec = excluded.duration_sec,
                status = excluded.status,
                updated_at = CURRENT_TIMESTAMP;
            """,
            (speaker_id, source_type, source_identifier, raw_path, duration_sec, status),
        )
        last_id = cursor.lastrowid or 0
    logger.debug(f"Recorded media entry in registry (ID: {last_id}) -> {raw_path}")
    return last_id


def update_media_status(
    raw_path: str,
    new_status: str,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Updates the pipeline processing state of a media item."""
    init_registry_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE media_registry SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE raw_path = ?;",
            (new_status, raw_path),
        )
        return cursor.rowcount > 0


def get_speaker_summary(
    speaker_id: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """Aggregates ingestion statistics grouped by speaker."""
    init_registry_db(db_path)
    query = """
        SELECT 
            speaker_id,
            COUNT(*) as total_files,
            SUM(duration_sec) as total_duration_sec,
            COUNT(CASE WHEN status = 'raw' THEN 1 END) as count_raw,
            COUNT(CASE WHEN status = 'enhanced' THEN 1 END) as count_enhanced,
            COUNT(CASE WHEN status = 'sliced' THEN 1 END) as count_sliced
        FROM media_registry
    """
    params: List[Any] = []
    if speaker_id:
        query += " WHERE speaker_id = ?"
        params.append(speaker_id)
    query += " GROUP BY speaker_id ORDER BY speaker_id ASC;"

    with get_db_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
