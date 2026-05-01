"""
prediction_store.py
===================
SQLite-backed durable store for the prediction log. Replaces the legacy
JSON file as the source of truth.

Why SQLite over JSON:
  - Atomic transactions: every log_prediction is one all-or-nothing
    write. A crash mid-write can't leave the store in a partial state.
  - WAL journal: SQLite's write-ahead log means even unclean shutdowns
    don't corrupt the database — committed transactions survive, and
    half-written ones roll back.
  - No silent truncation: a SELECT * on an empty database returns 0 rows.
    There's no "load returned [] because the file was malformed and the
    next write replaces 1,062 entries with 1" failure mode that wiped
    the JSON log on April 25, 2026.
  - Concurrent readers: multiple analytics endpoints can read in
    parallel without blocking each other or the writer.
  - Schema validation: types are enforced at write time, foreign keys
    cascade-delete cleanly, indexes make per-symbol queries cheap.

Schema (mirrors the v2 JSON shape):
  predictions               — one row per inference run
  prediction_horizons       — five rows per inference run (3D, 1W, 1M, 1Q, 1Y)
  prediction_scores         — one row per (horizon × scored interval)

Public API:
  init_db()                                  — create tables if missing
  insert_prediction(record)                  — upsert (dedupes by symbol+date)
  update_horizon_score(...)                  — write a per-interval score
  set_horizon_final(...)                     — mark horizon as fully scored
  get_all_predictions()                      — return list-of-dicts (legacy shape)
  get_predictions_by_symbol(symbol)          — filtered fetch
  count_predictions()                        — for health checks
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional


# ─── Paths ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(_HERE, ".predictions")
DB_PATH = os.path.join(DB_DIR, "prediction_log.db")


# ─── Schema ───────────────────────────────────────────────────────────────────
_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id  TEXT PRIMARY KEY,
    symbol         TEXT NOT NULL,
    timestamp      TEXT NOT NULL,            -- ISO 8601 with sub-second precision
    timestamp_date TEXT NOT NULL,            -- YYYY-MM-DD slice for dedup queries
    model_version  TEXT NOT NULL DEFAULT '1.0',
    regime         TEXT NOT NULL DEFAULT 'Unknown',
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date
    ON predictions(symbol, timestamp_date);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
    ON predictions(timestamp);

CREATE TABLE IF NOT EXISTS prediction_horizons (
    prediction_id     TEXT NOT NULL,
    horizon           TEXT NOT NULL,         -- "3 Day" | "1 Week" | ...
    predicted_return  REAL,
    predicted_price   REAL,
    current_price     REAL,
    confidence        REAL,
    ensemble_agreement REAL DEFAULT 0,
    val_dir_accuracy  REAL DEFAULT 0,
    direction         TEXT,
    top_features_json TEXT,                  -- JSON array of feature objs/strs
    final_scored      INTEGER NOT NULL DEFAULT 0,  -- 0/1
    final_correct     INTEGER,               -- 0/1 or NULL
    PRIMARY KEY (prediction_id, horizon),
    FOREIGN KEY (prediction_id)
        REFERENCES predictions(prediction_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_horizons_horizon
    ON prediction_horizons(horizon);

CREATE TABLE IF NOT EXISTS prediction_scores (
    prediction_id     TEXT NOT NULL,
    horizon           TEXT NOT NULL,
    interval          TEXT NOT NULL,         -- "1d" | "3d" | "7d" | ...
    actual_price      REAL,
    actual_return     REAL,
    direction_correct INTEGER,               -- 0/1 or NULL
    scored_at         TEXT,
    days_actual       INTEGER,
    PRIMARY KEY (prediction_id, horizon, interval),
    FOREIGN KEY (prediction_id, horizon)
        REFERENCES prediction_horizons(prediction_id, horizon)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_scores_interval
    ON prediction_scores(interval);
"""


# Connection-per-thread is the SQLite-recommended pattern. We don't use a
# pool; SQLite's WAL mode handles concurrency at the database level, not
# the connection level.
_local = threading.local()


def _connect() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    # WAL = writers don't block readers, partial writes never corrupt the
    # database, recovery on next open is automatic.
    conn.execute("PRAGMA journal_mode = WAL")
    # NORMAL = synchronous to disk on transaction commit, but lets some
    # OS caching happen between commits. Safe for our scale; FULL would
    # halve write throughput for marginal extra durability.
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def _conn():
    """
    Yields a thread-local connection. Commits on success, rolls back on
    exception. We don't close the connection — keeping it open avoids
    the WAL checkpoint cost on every call.
    """
    if not hasattr(_local, "conn"):
        _local.conn = _connect()
    c = _local.conn
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise


# ─── Init ─────────────────────────────────────────────────────────────────────
_init_lock = threading.Lock()
_initialized = False


def init_db() -> None:
    """
    Create tables if missing. Idempotent and cheap to call repeatedly —
    the second+ call is a no-op via the `_initialized` flag, so callers
    can sprinkle init_db() defensively without worrying about cost.
    """
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        with _conn() as c:
            c.executescript(_SCHEMA)
        _initialized = True


# ─── Writes ───────────────────────────────────────────────────────────────────

def _normalize_correct(v: Any) -> Optional[int]:
    """Convert Python bool/None to SQLite-friendly 0/1/NULL."""
    if v is True:
        return 1
    if v is False:
        return 0
    return None


def insert_prediction(record: Dict[str, Any]) -> str:
    """
    Insert (or replace, dedupe-style) a single prediction record.

    Mirrors the legacy log_prediction_v2 dedupe behavior: if a prediction
    already exists for the same (symbol, date), we replace it AFTER
    carrying over any scores already recorded so we don't lose grading
    progress. Single transaction — caller never sees a half-written
    state, even if the process crashes mid-flight.
    """
    init_db()

    pred_id = record["prediction_id"]
    symbol = record["symbol"]
    timestamp = record["timestamp"]
    timestamp_date = timestamp[:10]

    with _conn() as c:
        # ── Carry-over: find any existing entry for (symbol, date) ─────
        # We collect its scores first, before deletion, then re-apply
        # them to the new horizons after the new record is inserted.
        existing_row = c.execute(
            "SELECT prediction_id FROM predictions "
            "WHERE symbol = ? AND timestamp_date = ?",
            (symbol, timestamp_date),
        ).fetchone()

        carry_over: Dict[str, Dict[str, Dict[str, Any]]] = {}  # {horizon: {interval: row}}
        carry_final: Dict[str, Optional[bool]] = {}  # {horizon: final_correct}

        if existing_row:
            existing_id = existing_row["prediction_id"]
            for s in c.execute(
                "SELECT horizon, interval, actual_price, actual_return, "
                "direction_correct, scored_at, days_actual "
                "FROM prediction_scores WHERE prediction_id = ?",
                (existing_id,),
            ):
                carry_over.setdefault(s["horizon"], {})[s["interval"]] = dict(s)
            for h in c.execute(
                "SELECT horizon, final_scored, final_correct "
                "FROM prediction_horizons WHERE prediction_id = ?",
                (existing_id,),
            ):
                if h["final_scored"]:
                    carry_final[h["horizon"]] = (
                        None if h["final_correct"] is None
                        else bool(h["final_correct"])
                    )

            c.execute(
                "DELETE FROM predictions WHERE prediction_id = ?",
                (existing_id,),
            )

        # ── Insert the new prediction ───────────────────────────────────
        c.execute(
            "INSERT INTO predictions "
            "(prediction_id, symbol, timestamp, timestamp_date, model_version, regime) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                pred_id,
                symbol,
                timestamp,
                timestamp_date,
                record.get("model_version") or "1.0",
                record.get("regime") or "Unknown",
            ),
        )

        for horizon, hdata in (record.get("horizons") or {}).items():
            top_features = hdata.get("top_features") or []
            top_features_json = json.dumps(top_features, default=str)
            final_scored = bool(hdata.get("final_scored"))
            final_correct = _normalize_correct(hdata.get("final_correct"))

            # If the OLD entry had a populated final_correct for this
            # horizon, prefer it (matches the legacy carry-over rule).
            if horizon in carry_final and not final_scored:
                old = carry_final[horizon]
                final_scored = True
                final_correct = _normalize_correct(old)

            c.execute(
                "INSERT INTO prediction_horizons "
                "(prediction_id, horizon, predicted_return, predicted_price, "
                " current_price, confidence, ensemble_agreement, val_dir_accuracy, "
                " direction, top_features_json, final_scored, final_correct) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pred_id,
                    horizon,
                    hdata.get("predicted_return"),
                    hdata.get("predicted_price"),
                    hdata.get("current_price"),
                    hdata.get("confidence"),
                    hdata.get("ensemble_agreement") or 0,
                    hdata.get("val_dir_accuracy") or 0,
                    hdata.get("direction"),
                    top_features_json,
                    1 if final_scored else 0,
                    final_correct,
                ),
            )

            # Insert the new prediction's own scores (usually empty for
            # a fresh log entry — scoring fills in later)
            new_scores = (hdata.get("scores") or {})
            # Then merge in any carry-over scores from the old entry
            # for the same horizon, but only intervals not already
            # populated by the new one.
            merged = dict(carry_over.get(horizon, {}))
            for interval, sdata in new_scores.items():
                merged[interval] = sdata

            for interval, sdata in merged.items():
                c.execute(
                    "INSERT INTO prediction_scores "
                    "(prediction_id, horizon, interval, actual_price, "
                    " actual_return, direction_correct, scored_at, days_actual) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        pred_id,
                        horizon,
                        interval,
                        sdata.get("actual_price"),
                        sdata.get("actual_return"),
                        _normalize_correct(sdata.get("direction_correct")),
                        sdata.get("scored_at"),
                        sdata.get("days_actual"),
                    ),
                )

    return pred_id


def update_horizon_score(
    prediction_id: str,
    horizon: str,
    interval: str,
    actual_price: float,
    actual_return: float,
    direction_correct: bool,
    scored_at: str,
    days_actual: int,
) -> None:
    """Insert or replace a single per-interval score entry."""
    init_db()
    with _conn() as c:
        c.execute(
            "INSERT INTO prediction_scores "
            "(prediction_id, horizon, interval, actual_price, "
            " actual_return, direction_correct, scored_at, days_actual) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(prediction_id, horizon, interval) DO UPDATE SET "
            " actual_price = excluded.actual_price, "
            " actual_return = excluded.actual_return, "
            " direction_correct = excluded.direction_correct, "
            " scored_at = excluded.scored_at, "
            " days_actual = excluded.days_actual",
            (
                prediction_id, horizon, interval,
                actual_price, actual_return,
                _normalize_correct(direction_correct),
                scored_at, days_actual,
            ),
        )


def set_horizon_final(
    prediction_id: str,
    horizon: str,
    final_correct: Optional[bool],
) -> None:
    """Mark a horizon as fully scored at its expiration."""
    init_db()
    with _conn() as c:
        c.execute(
            "UPDATE prediction_horizons SET "
            " final_scored = 1, final_correct = ? "
            "WHERE prediction_id = ? AND horizon = ?",
            (_normalize_correct(final_correct), prediction_id, horizon),
        )


# ─── Reads ────────────────────────────────────────────────────────────────────

def _row_to_horizon_dict(h_row: sqlite3.Row, scores_for_h: Dict[str, sqlite3.Row]) -> Dict[str, Any]:
    """Convert a horizon row + its scores into the legacy dict shape."""
    try:
        top_features = json.loads(h_row["top_features_json"] or "[]")
    except (json.JSONDecodeError, TypeError):
        top_features = []

    scores: Dict[str, Dict[str, Any]] = {}
    for iv, s in scores_for_h.items():
        scores[iv] = {
            "actual_price":      s["actual_price"],
            "actual_return":     s["actual_return"],
            "direction_correct": (
                None if s["direction_correct"] is None
                else bool(s["direction_correct"])
            ),
            "scored_at":         s["scored_at"],
            "days_actual":       s["days_actual"],
        }

    return {
        "predicted_return":   h_row["predicted_return"] or 0,
        "predicted_price":    h_row["predicted_price"] or 0,
        "current_price":      h_row["current_price"] or 0,
        "confidence":         h_row["confidence"] or 0,
        "ensemble_agreement": h_row["ensemble_agreement"] or 0,
        "val_dir_accuracy":   h_row["val_dir_accuracy"] or 0,
        "direction":          h_row["direction"] or "",
        "top_features":       top_features,
        "scores":             scores,
        "final_scored":       bool(h_row["final_scored"]),
        "final_correct":      (
            None if h_row["final_correct"] is None
            else bool(h_row["final_correct"])
        ),
    }


def get_all_predictions() -> List[Dict[str, Any]]:
    """
    Return every prediction in the same shape the JSON file used to
    return — a list of dicts, each with `prediction_id`, `symbol`,
    `timestamp`, `model_version`, `regime`, and a `horizons` dict.

    All higher-level analytics code in prediction_logger_v2 consumes
    this shape, so keeping it stable means nothing else has to change
    when we swap the storage layer.
    """
    init_db()
    with _conn() as c:
        # Pull predictions, horizons, scores in three queries.
        # For the typical 1k-10k row scale this is well under 100ms.
        preds_rows = list(c.execute(
            "SELECT prediction_id, symbol, timestamp, model_version, regime "
            "FROM predictions ORDER BY timestamp"
        ))
        horizons_rows = list(c.execute(
            "SELECT prediction_id, horizon, predicted_return, predicted_price, "
            "       current_price, confidence, ensemble_agreement, val_dir_accuracy, "
            "       direction, top_features_json, final_scored, final_correct "
            "FROM prediction_horizons"
        ))
        scores_rows = list(c.execute(
            "SELECT prediction_id, horizon, interval, actual_price, "
            "       actual_return, direction_correct, scored_at, days_actual "
            "FROM prediction_scores"
        ))

    # Group scores by (prediction_id, horizon)
    scores_by_h: Dict[tuple, Dict[str, sqlite3.Row]] = {}
    for s in scores_rows:
        key = (s["prediction_id"], s["horizon"])
        scores_by_h.setdefault(key, {})[s["interval"]] = s

    # Group horizons by prediction_id
    horizons_by_pid: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for h in horizons_rows:
        pid = h["prediction_id"]
        scores_for_h = scores_by_h.get((pid, h["horizon"]), {})
        horizons_by_pid.setdefault(pid, {})[h["horizon"]] = _row_to_horizon_dict(h, scores_for_h)

    out: List[Dict[str, Any]] = []
    for p in preds_rows:
        out.append({
            "prediction_id":  p["prediction_id"],
            "symbol":         p["symbol"],
            "timestamp":      p["timestamp"],
            "model_version":  p["model_version"],
            "regime":         p["regime"],
            "horizons":       horizons_by_pid.get(p["prediction_id"], {}),
        })
    return out


def find_todays_prediction(symbol: str) -> Optional[str]:
    """
    Anchor § 8.1 same-day dedupe lookup (file/SQLite path).

    Return the most-recent prediction_id for `symbol` whose timestamp falls
    on the current local date, or None if none exists. The legacy SQLite
    schema doesn't carry user_id and stores all five horizons under one
    prediction_id, so the dedupe granularity here is per-symbol per-day —
    broader than the Supabase-mode dedupe (which is per-user per-symbol
    per-horizon per-day). That's acceptable because file mode is dev-only;
    production runs on Supabase.
    """
    init_db()
    sym = (symbol or "").upper().strip()
    if not sym:
        return None
    today = datetime.now().date().isoformat()
    with _conn() as c:
        row = c.execute(
            "SELECT prediction_id FROM predictions "
            "WHERE symbol = ? AND timestamp_date = ? "
            "ORDER BY timestamp ASC LIMIT 1",
            (sym, today),
        ).fetchone()
    return row["prediction_id"] if row else None


def get_predictions_by_symbol(symbol: str) -> List[Dict[str, Any]]:
    """Same shape as get_all_predictions but filtered to one symbol."""
    init_db()
    sym = (symbol or "").upper().strip()
    with _conn() as c:
        preds_rows = list(c.execute(
            "SELECT prediction_id, symbol, timestamp, model_version, regime "
            "FROM predictions WHERE symbol = ? ORDER BY timestamp",
            (sym,),
        ))
        if not preds_rows:
            return []
        pids = tuple(p["prediction_id"] for p in preds_rows)
        placeholders = ",".join("?" * len(pids))
        horizons_rows = list(c.execute(
            f"SELECT * FROM prediction_horizons WHERE prediction_id IN ({placeholders})",
            pids,
        ))
        scores_rows = list(c.execute(
            f"SELECT * FROM prediction_scores WHERE prediction_id IN ({placeholders})",
            pids,
        ))

    scores_by_h: Dict[tuple, Dict[str, sqlite3.Row]] = {}
    for s in scores_rows:
        scores_by_h.setdefault((s["prediction_id"], s["horizon"]), {})[s["interval"]] = s

    horizons_by_pid: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for h in horizons_rows:
        scores_for_h = scores_by_h.get((h["prediction_id"], h["horizon"]), {})
        horizons_by_pid.setdefault(h["prediction_id"], {})[h["horizon"]] = (
            _row_to_horizon_dict(h, scores_for_h)
        )

    return [
        {
            "prediction_id":  p["prediction_id"],
            "symbol":         p["symbol"],
            "timestamp":      p["timestamp"],
            "model_version":  p["model_version"],
            "regime":         p["regime"],
            "horizons":       horizons_by_pid.get(p["prediction_id"], {}),
        }
        for p in preds_rows
    ]


def count_predictions() -> int:
    """Total prediction-run count (not horizon-level)."""
    init_db()
    with _conn() as c:
        row = c.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()
        return int(row["n"]) if row else 0


def health_check() -> Dict[str, Any]:
    """Quick stats for diagnostics + startup checks."""
    init_db()
    with _conn() as c:
        n_preds = c.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]
        n_horizons = c.execute("SELECT COUNT(*) AS n FROM prediction_horizons").fetchone()["n"]
        n_scores = c.execute("SELECT COUNT(*) AS n FROM prediction_scores").fetchone()["n"]
        first = c.execute(
            "SELECT timestamp FROM predictions ORDER BY timestamp ASC LIMIT 1"
        ).fetchone()
        last = c.execute(
            "SELECT timestamp FROM predictions ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    return {
        "predictions":  n_preds,
        "horizons":     n_horizons,
        "scores":       n_scores,
        "earliest":     first["timestamp"] if first else None,
        "latest":       last["timestamp"] if last else None,
        "db_path":      DB_PATH,
    }
