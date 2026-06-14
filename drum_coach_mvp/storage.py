from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "drum_coach_sessions.db"


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection; callers should use it in a with-statement."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                file_name TEXT NOT NULL,
                target_bpm REAL,
                practice_type TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                estimated_bpm REAL NOT NULL,
                timing_stability_score REAL NOT NULL,
                tempo_drift_bpm REAL NOT NULL,
                dynamics_consistency_score REAL NOT NULL,
                overall_score REAL NOT NULL,
                tips_json TEXT NOT NULL
            )
            """
        )


def save_session(result: dict[str, Any]) -> None:
    init_db()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO sessions (
                file_name,
                target_bpm,
                practice_type,
                duration_seconds,
                estimated_bpm,
                timing_stability_score,
                tempo_drift_bpm,
                dynamics_consistency_score,
                overall_score,
                tips_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["file_name"],
                result.get("target_bpm"),
                result["practice_type"],
                result["duration_seconds"],
                result["estimated_bpm"],
                result["timing_stability_score"],
                result["tempo_drift_bpm"],
                result["dynamics_consistency_score"],
                result["overall_score"],
                json.dumps(result["coaching_tips"]),
            ),
        )


def load_sessions(practice_type: str | None = None) -> pd.DataFrame:
    init_db()
    query = """
        SELECT
            created_at,
            file_name,
            practice_type,
            target_bpm,
            duration_seconds,
            estimated_bpm,
            timing_stability_score,
            tempo_drift_bpm,
            dynamics_consistency_score,
            overall_score,
            tips_json
        FROM sessions
    """
    params: tuple[Any, ...] = ()
    if practice_type and practice_type.lower() != "all":
        query += " WHERE practice_type = ?"
        params = (practice_type,)
    query += " ORDER BY datetime(created_at) DESC, id DESC"

    with get_connection() as connection:
        frame = pd.read_sql_query(query, connection, params=params)

    if frame.empty:
        return frame

    frame["created_at"] = pd.to_datetime(frame["created_at"])
    frame["coaching_tips"] = frame["tips_json"].apply(json.loads)
    return frame.drop(columns=["tips_json"])
