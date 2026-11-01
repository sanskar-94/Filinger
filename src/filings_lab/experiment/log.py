"""Experiment event log: exposures (one per answered request) and feedback (thumbs up or down).

SQLite keeps the lab self contained; the schema is plain SQL so the same statements run on Postgres in a
deployed setup. ``export`` writes Parquet and CSV extracts for the BI dashboard.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS exposures (
    request_id TEXT PRIMARY KEY, ts REAL NOT NULL, experiment TEXT NOT NULL, variant TEXT NOT NULL,
    user_id TEXT NOT NULL, ticker TEXT, mode TEXT, model_version TEXT, latency_s REAL, n_citations INTEGER,
    citation_valid_rate REAL, cost_usd REAL, error TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    request_id TEXT PRIMARY KEY REFERENCES exposures(request_id), ts REAL NOT NULL,
    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)), comment TEXT
);
CREATE INDEX IF NOT EXISTS ix_exposures_exp ON exposures(experiment, variant);
"""


class EventLog:
    def __init__(self, path: str | Path = "data/experiments.db"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(exposures)")}
            if "model_version" not in cols:  # migrate databases created before the column existed
                c.execute("ALTER TABLE exposures ADD COLUMN model_version TEXT")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    def exposure(self, **row) -> None:
        row.setdefault("ts", time.time())
        cols = ", ".join(row)
        with self.lock, self._conn() as c:
            c.execute(f"INSERT OR REPLACE INTO exposures ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))

    def feedback(self, request_id: str, rating: int, comment: str | None = None, ts: float | None = None) -> bool:
        with self.lock, self._conn() as c:
            if not c.execute("SELECT 1 FROM exposures WHERE request_id = ?", (request_id,)).fetchone():
                return False
            c.execute("INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?)",
                      (request_id, ts if ts is not None else time.time(), int(rating), comment))
        return True

    def frame(self, experiment: str | None = None) -> pd.DataFrame:
        q = """SELECT e.*, f.rating, f.ts AS feedback_ts FROM exposures e LEFT JOIN feedback f USING (request_id)"""
        params: tuple = ()
        if experiment:
            q += " WHERE e.experiment = ?"
            params = (experiment,)
        with self._conn() as c:
            return pd.read_sql_query(q, c, params=params)
