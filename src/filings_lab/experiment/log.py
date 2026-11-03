"""Experiment event log: exposures (one per answered request) and feedback (thumbs up or down).

Each API replica appends to its own SQLite file, so writes stay local and fast. The service ships a consistent
snapshot of that file to object storage on a timer and at shutdown (``snapshot``), and the analysis reads the
union of every replica's file (``load_frame``). A rating can reach a different replica than the one that logged
its exposure; the request id carries an HMAC so any replica can accept it, and the join happens at analysis
time. The schema is plain SQL, so the same statements run on Postgres if the log moves to a shared database.
``export`` writes Parquet and CSV extracts for the BI dashboard.
"""
from __future__ import annotations

import glob
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd

from .. import storage

SCHEMA = """
CREATE TABLE IF NOT EXISTS exposures (
    request_id TEXT PRIMARY KEY, ts REAL NOT NULL, experiment TEXT NOT NULL, variant TEXT NOT NULL,
    user_id TEXT NOT NULL, ticker TEXT, mode TEXT, model_version TEXT, latency_s REAL, n_citations INTEGER,
    citation_valid_rate REAL, cost_usd REAL, error TEXT, n_extractions INTEGER, n_extraction_fallbacks INTEGER
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
            for col, typ in (("model_version", "TEXT"), ("n_extractions", "INTEGER"), ("n_extraction_fallbacks", "INTEGER")):
                if col not in cols:  # migrate databases created before the column existed
                    c.execute(f"ALTER TABLE exposures ADD COLUMN {col} {typ}")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    def exposure(self, **row) -> None:
        row.setdefault("ts", time.time())
        cols = ", ".join(row)
        with self.lock, self._conn() as c:
            c.execute(f"INSERT OR REPLACE INTO exposures ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))

    def feedback(self, request_id: str, rating: int, comment: str | None = None, ts: float | None = None,
                 known: bool = False) -> bool:
        """Record a rating. ``known`` means the caller verified the request id (its signature), so the exposure
        may sit in another replica's log; otherwise the exposure must be in this one."""
        with self.lock, self._conn() as c:
            if not known and not c.execute("SELECT 1 FROM exposures WHERE request_id = ?", (request_id,)).fetchone():
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

    def snapshot(self, dest: str) -> None:
        """Write a consistent copy of the live database (SQLite online backup, safe while requests are being
        logged) to a local path or an s3:// URI."""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / "snapshot.db"
            src, dst = self._conn(), sqlite3.connect(tmp)
            try:
                with self.lock:
                    src.backup(dst)
            finally:
                dst.close()
                src.close()
            storage.write_bytes(dest, tmp.read_bytes())


def resolve_sources(spec: str) -> list[str]:
    """``--db`` value to database files: a path, a comma separated list, a local glob, or an s3:// prefix
    ending in / (every .db under it, one per replica)."""
    out: list[str] = []
    for part in (x.strip() for x in str(spec).split(",")):
        if not part:
            continue
        if storage.is_remote(part) and part.endswith("/"):
            out += storage.list_uris(part, suffix=".db")
        elif any(ch in part for ch in "*?["):
            out += sorted(glob.glob(part))
        else:
            out.append(part)
    if not out:
        raise FileNotFoundError(f"no event databases match {spec}")
    return out


def load_frame(spec: str, experiment: str | None = None) -> pd.DataFrame:
    """Exposures joined to feedback across every replica's log. Same columns as ``EventLog.frame``."""
    sources = resolve_sources(spec)
    if len(sources) == 1 and not storage.is_remote(sources[0]):
        return EventLog(sources[0]).frame(experiment)
    exposures, feedback = [], []
    with tempfile.TemporaryDirectory() as d:
        for i, uri in enumerate(sources):
            path = uri
            if storage.is_remote(uri):
                path = str(Path(d) / f"replica-{i}.db")
                Path(path).write_bytes(storage.read_bytes(uri))
            with sqlite3.connect(path) as c:
                exposures.append(pd.read_sql_query("SELECT * FROM exposures", c))
                feedback.append(pd.read_sql_query("SELECT request_id, ts, rating FROM feedback", c))
    e = pd.concat(exposures, ignore_index=True).sort_values("ts").drop_duplicates("request_id", keep="last")
    if experiment:
        e = e[e["experiment"] == experiment]
    f = (pd.concat(feedback, ignore_index=True).sort_values("ts").drop_duplicates("request_id", keep="last")
         .rename(columns={"ts": "feedback_ts"}))
    out = e.merge(f[["request_id", "rating", "feedback_ts"]], on="request_id", how="left")
    return out.reset_index(drop=True)
