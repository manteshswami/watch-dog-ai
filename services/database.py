"""
WatchAI — FrameIndexer (SQLite + ChromaDB)
Dual-layer persistence for CCTV surveillance events:
  - SQLite  : fast structured queries (by time, location, threat level, criminal name)
  - ChromaDB: semantic vector search over natural language event descriptions

Replaces the old single-table alerts.db with a richer schema.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb

import config
from services.embeddings import get_embeddings

logger = logging.getLogger(__name__)


# ── SQLite Schema ─────────────────────────────────────────────────────────────

CREATE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS frames (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id        INTEGER NOT NULL,
    timestamp       TEXT    NOT NULL,
    location        TEXT    NOT NULL,
    camera_id       TEXT    NOT NULL DEFAULT 'CAM-01',
    objects_json    TEXT    NOT NULL,
    detections_json TEXT,
    vlm_description TEXT,
    threat_level    TEXT    DEFAULT 'LOW',
    criminal_name   TEXT,
    risk_score      INTEGER DEFAULT 0,
    risk_level      TEXT    DEFAULT 'LOW',
    alert_triggered INTEGER DEFAULT 0,
    frame_path      TEXT,
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id    INTEGER,
    timestamp   TEXT    NOT NULL,
    location    TEXT    NOT NULL,
    camera_id   TEXT    NOT NULL DEFAULT 'CAM-01',
    rule_name   TEXT    NOT NULL,
    alert_text  TEXT    NOT NULL,
    severity    TEXT    NOT NULL,
    criminal    TEXT,
    created_at  TEXT    DEFAULT (datetime('now'))
);
"""


@contextmanager
def _get_conn(db_path: str = None):
    if db_path is None:
        db_path = config.SQLITE_DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_db(db_path: str = None) -> None:
    if db_path is None:
        db_path = config.SQLITE_DB_PATH
    with _get_conn(db_path) as conn:
        conn.executescript(CREATE_SCHEMA_SQL)
    logger.info("SQLite database initialised.")


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def _get_chroma_collection():
    """Return (or create) the ChromaDB collection with Ollama embeddings."""
    import os
    # Ensure the persistence directory exists
    os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
    # Explicitly specify a tenant to avoid "default_tenant" connection errors
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR, tenant="default_tenant")
    ef = get_embeddings(config.EMBEDDING_MODEL)
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ── FrameIndexer ──────────────────────────────────────────────────────────────

class FrameIndexer:
    """Stores and retrieves CCTV surveillance events from SQLite + ChromaDB.

    Drop-in replacement for the old services/database.py.
    """

    def __init__(self):
        _init_db()
        self._collection = _get_chroma_collection()

    # ── Write ────────────────────────────────────────────────────────────────

    def index_event(self, event: Dict[str, Any]) -> int:
        """
        Persist a merged surveillance event to SQLite + ChromaDB.
        Returns the SQLite row id.

        Expected event keys:
            frame_id, timestamp, location, camera_id,
            objects, detections, vlm_description, threat_level,
            criminal_name, risk_score, risk_level, frame_path
        """
        objects     = event.get("objects", [])
        detections  = event.get("detections", [])
        criminal    = event.get("criminal_name") or ""
        risk_score  = event.get("risk_score", 0)
        risk_level  = event.get("risk_level", "LOW")

        # ── SQLite ────────────────────────────────────────────────────────
        with _get_conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO frames
                    (frame_id, timestamp, location, camera_id,
                     objects_json, detections_json, vlm_description, threat_level,
                     criminal_name, risk_score, risk_level,
                     alert_triggered, frame_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.get("frame_id", 0),
                    event.get("timestamp", ""),
                    event.get("location", config.DEFAULT_LOCATION),
                    event.get("camera_id", "CAM-01"),
                    json.dumps(objects),
                    json.dumps(detections),
                    event.get("vlm_description", ""),
                    event.get("threat_level", "LOW"),
                    criminal,
                    risk_score,
                    risk_level,
                    int(bool(event.get("alert_triggered", False))),
                    event.get("frame_path", ""),
                ),
            )
            row_id = cur.lastrowid

        # ── ChromaDB ─────────────────────────────────────────────────────
        doc_text = (
            f"Location: {event.get('location', '')}. "
            f"Objects: {', '.join(objects) if objects else 'none'}. "
            f"Description: {event.get('vlm_description', '')}. "
            f"Threat: {event.get('threat_level', 'LOW')}. "
            + (f"Criminal: {criminal}. " if criminal else "")
        )
        self._collection.upsert(
            ids=[f"frame_{row_id}"],
            documents=[doc_text],
            metadatas=[{
                "frame_id":    str(event.get("frame_id", 0)),
                "timestamp":   event.get("timestamp", ""),
                "location":    event.get("location", config.DEFAULT_LOCATION),
                "threat_level": event.get("threat_level", "LOW"),
                "objects":     ",".join(objects),
                "criminal":    criminal,
                "camera_id":   event.get("camera_id", "CAM-01"),
            }],
        )
        logger.debug(f"Event indexed — SQLite row {row_id}")
        return row_id

    def log_alert(self, frame_row_id: int, alert: Dict[str, Any]) -> None:
        """Persist a triggered alert to SQLite."""
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO alerts
                    (frame_id, timestamp, location, camera_id,
                     rule_name, alert_text, severity, criminal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    frame_row_id,
                    alert.get("timestamp", ""),
                    alert.get("location", ""),
                    alert.get("camera_id", "CAM-01"),
                    alert.get("rule_name", ""),
                    alert.get("alert_text", ""),
                    alert.get("severity", "LOW"),
                    alert.get("criminal", ""),
                ),
            )

    # ── Read (SQLite) ────────────────────────────────────────────────────────

    def query_by_time(self, start: str, end: str) -> List[Dict]:
        """Return events between two ISO timestamps."""
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM frames WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
                (start, end),
            ).fetchall()
        return [dict(r) for r in rows]

    def query_by_object(self, object_type: str) -> List[Dict]:
        """Return events where objects_json contains the given label."""
        with _get_conn() as conn:
            rows = conn.execute(
                'SELECT * FROM frames WHERE objects_json LIKE ? ORDER BY timestamp DESC',
                (f'%"{object_type}"%',),
            ).fetchall()
        return [dict(r) for r in rows]

    def query_by_criminal(self, name: str) -> List[Dict]:
        """Return all events where a named criminal was matched."""
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM frames WHERE criminal_name LIKE ? ORDER BY timestamp DESC",
                (f"%{name.lower()}%",),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_alerts(self, severity: Optional[str] = None,
                       criminal: Optional[str] = None) -> List[Dict]:
        """Return alerts, optionally filtered by severity and/or criminal name."""
        with _get_conn() as conn:
            clauses, params = [], []
            if severity:
                clauses.append("severity = ?")
                params.append(severity)
            if criminal:
                clauses.append("criminal LIKE ?")
                params.append(f"%{criminal.lower()}%")
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            rows = conn.execute(
                f"SELECT * FROM alerts {where} ORDER BY timestamp DESC",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_frames(self, limit: int = 200) -> List[Dict]:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM frames ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_summary(self, date: Optional[str] = None) -> Dict:
        """Aggregate counts for a given date (YYYY-MM-DD), defaults to today."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        with _get_conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM frames WHERE timestamp LIKE ?", (f"{date}%",)
            ).fetchone()[0]
            total_alerts = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE timestamp LIKE ?", (f"{date}%",)
            ).fetchone()[0]
            high_threats = conn.execute(
                "SELECT COUNT(*) FROM frames WHERE threat_level='HIGH' AND timestamp LIKE ?",
                (f"{date}%",),
            ).fetchone()[0]
            criminal_hits = conn.execute(
                "SELECT COUNT(*) FROM frames WHERE criminal_name != '' AND timestamp LIKE ?",
                (f"{date}%",),
            ).fetchone()[0]
            obj_rows = conn.execute(
                "SELECT objects_json FROM frames WHERE timestamp LIKE ?", (f"{date}%",)
            ).fetchall()

        object_counts: Dict[str, int] = {}
        for row in obj_rows:
            for obj in json.loads(row[0]):
                object_counts[obj] = object_counts.get(obj, 0) + 1

        return {
            "date":           date,
            "total_frames":   total,
            "total_alerts":   total_alerts,
            "high_threat_frames": high_threats,
            "criminal_matches": criminal_hits,
            "object_counts":  object_counts,
        }

    # ── Read (ChromaDB semantic) ──────────────────────────────────────────────

    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Natural language semantic search over indexed frame descriptions."""
        try:
            count = self._collection.count()
            if count == 0:
                return []
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, count),
            )
            docs  = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            return [
                {"document": doc, "metadata": meta}
                for doc, meta in zip(docs, metas)
            ]
        except Exception as exc:
            logger.error(f"ChromaDB query failed: {exc}")
            return []


# ── Legacy shim (keeps old log_alert() call working in main.py during migration) ──

_indexer: Optional[FrameIndexer] = None

def init_db():
    global _indexer
    _indexer = FrameIndexer()

def log_alert(name: str, timestamp: str, confidence: float,
              risk_score: int = 0, risk_level: str = "UNKNOWN",
              location: str = ""):
    """Legacy shim for backward compatibility."""
    if _indexer is None:
        init_db()
    alert = {
        "timestamp":  timestamp,
        "location":   location or config.DEFAULT_LOCATION,
        "camera_id":  "CAM-01",
        "rule_name":  "criminal_face_match",
        "alert_text": (
            f"Criminal '{name}' identified. "
            f"Risk: {risk_level} (score={risk_score}), "
            f"conf={confidence:.2f}"
        ),
        "severity":   "CRITICAL" if risk_level == "CRITICAL" else "HIGH",
        "criminal":   name,
    }
    _indexer.log_alert(0, alert)

def get_recent_alerts(limit: int = 20) -> list:
    """Legacy shim."""
    if _indexer is None:
        init_db()
    return _indexer.get_all_alerts()[:limit]
