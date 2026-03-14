# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SQLite-backed re-identification embedding store.

Stores appearance feature vectors for detected targets (people, vehicles)
and performs cosine similarity search to find cross-camera matches without
facial recognition.  Uses numpy for fast vector math when available, with
a pure-Python fallback for constrained environments.
"""

from __future__ import annotations

import math
import sqlite3
import struct
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional numpy — pure-Python fallback for cosine similarity
# ---------------------------------------------------------------------------

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Vector serialization helpers
# ---------------------------------------------------------------------------

def _vector_to_blob(vec: list[float]) -> bytes:
    """Pack a float vector into a compact binary blob (little-endian float32)."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _blob_to_vector(blob: bytes) -> list[float]:
    """Unpack a binary blob back into a float vector."""
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine_similarity_numpy(a: list[float], b: list[float]) -> float:
    """Cosine similarity using numpy."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    dot = float(np.dot(va, vb))
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cosine_similarity_pure(a: list[float], b: list[float]) -> float:
    """Cosine similarity in pure Python."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Uses numpy when available, otherwise falls back to pure Python.
    """
    if _HAS_NUMPY:
        return _cosine_similarity_numpy(a, b)
    return _cosine_similarity_pure(a, b)


# ---------------------------------------------------------------------------
# SQL schemas
# ---------------------------------------------------------------------------

_SCHEMA_EMBEDDINGS = """
CREATE TABLE IF NOT EXISTS reid_embeddings (
    embedding_id TEXT PRIMARY KEY,
    target_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    source_camera TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    confidence REAL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_reid_target ON reid_embeddings(target_id);
CREATE INDEX IF NOT EXISTS idx_reid_camera ON reid_embeddings(source_camera);
CREATE INDEX IF NOT EXISTS idx_reid_ts ON reid_embeddings(timestamp);
"""

_SCHEMA_MATCHES = """
CREATE TABLE IF NOT EXISTS reid_matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_embedding_id TEXT NOT NULL,
    matched_embedding_id TEXT NOT NULL,
    similarity REAL NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (query_embedding_id) REFERENCES reid_embeddings(embedding_id),
    FOREIGN KEY (matched_embedding_id) REFERENCES reid_embeddings(embedding_id)
);
CREATE INDEX IF NOT EXISTS idx_match_query ON reid_matches(query_embedding_id);
CREATE INDEX IF NOT EXISTS idx_match_matched ON reid_matches(matched_embedding_id);
"""


def _utcnow() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# ReIDStore
# ---------------------------------------------------------------------------

class ReIDStore:
    """SQLite-backed re-identification embedding store.

    Stores appearance feature vectors and performs cosine similarity search
    to find cross-camera matches.  Thread-safe via a lock around writes.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``":memory:"`` for an
        ephemeral in-memory database (useful for testing).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(_SCHEMA_EMBEDDINGS)
        cur.executescript(_SCHEMA_MATCHES)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Embedding storage
    # ------------------------------------------------------------------

    def store_embedding(
        self,
        target_id: str,
        embedding_vector: list[float],
        source_camera: str,
        confidence: float = 1.0,
        embedding_id: Optional[str] = None,
    ) -> str:
        """Save a feature vector for a detected target.

        Parameters
        ----------
        target_id:
            Identifier for the tracked target (e.g. detection tracking ID).
        embedding_vector:
            The appearance feature vector (typically 128-2048 floats).
        source_camera:
            Camera source ID that captured this detection.
        confidence:
            Detection confidence score (0.0 to 1.0).
        embedding_id:
            Optional explicit ID.  If ``None``, one is generated from
            target_id + timestamp.

        Returns the embedding_id of the stored record.
        """
        import hashlib

        ts = _utcnow()
        if embedding_id is None:
            raw = f"{target_id}:{source_camera}:{ts}"
            embedding_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

        blob = _vector_to_blob(embedding_vector)

        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO reid_embeddings
                   (embedding_id, target_id, embedding, source_camera, timestamp, confidence)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (embedding_id, target_id, blob, source_camera, ts, confidence),
            )
            self._conn.commit()

        return embedding_id

    def get_embedding(self, embedding_id: str) -> Optional[dict]:
        """Retrieve a single embedding by ID.

        Returns a dict with ``embedding_id``, ``target_id``, ``embedding``
        (as list[float]), ``source_camera``, ``timestamp``, ``confidence``,
        or ``None`` if not found.
        """
        row = self._conn.execute(
            """SELECT embedding_id, target_id, embedding, source_camera,
                      timestamp, confidence
               FROM reid_embeddings WHERE embedding_id = ?""",
            (embedding_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "embedding_id": row["embedding_id"],
            "target_id": row["target_id"],
            "embedding": _blob_to_vector(row["embedding"]),
            "source_camera": row["source_camera"],
            "timestamp": row["timestamp"],
            "confidence": row["confidence"],
        }

    def get_embeddings_for_target(self, target_id: str) -> list[dict]:
        """Get all embeddings for a specific target ID."""
        rows = self._conn.execute(
            """SELECT embedding_id, target_id, embedding, source_camera,
                      timestamp, confidence
               FROM reid_embeddings
               WHERE target_id = ?
               ORDER BY timestamp DESC""",
            (target_id,),
        ).fetchall()
        return [
            {
                "embedding_id": r["embedding_id"],
                "target_id": r["target_id"],
                "embedding": _blob_to_vector(r["embedding"]),
                "source_camera": r["source_camera"],
                "timestamp": r["timestamp"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def find_similar(
        self,
        embedding_vector: list[float],
        threshold: float = 0.7,
        limit: int = 10,
    ) -> list[dict]:
        """Find stored embeddings similar to the query vector.

        Performs a brute-force cosine similarity search against all stored
        embeddings.  Returns results sorted by descending similarity.

        Parameters
        ----------
        embedding_vector:
            The query feature vector.
        threshold:
            Minimum cosine similarity to include in results (0.0 to 1.0).
        limit:
            Maximum number of results to return.

        Returns a list of dicts with ``embedding_id``, ``target_id``,
        ``source_camera``, ``timestamp``, ``confidence``, and ``similarity``.
        """
        rows = self._conn.execute(
            """SELECT embedding_id, target_id, embedding, source_camera,
                      timestamp, confidence
               FROM reid_embeddings"""
        ).fetchall()

        results: list[dict] = []
        for row in rows:
            stored_vec = _blob_to_vector(row["embedding"])
            if len(stored_vec) != len(embedding_vector):
                continue
            sim = cosine_similarity(embedding_vector, stored_vec)
            if sim >= threshold:
                results.append({
                    "embedding_id": row["embedding_id"],
                    "target_id": row["target_id"],
                    "source_camera": row["source_camera"],
                    "timestamp": row["timestamp"],
                    "confidence": row["confidence"],
                    "similarity": round(sim, 6),
                })

        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Match recording and retrieval
    # ------------------------------------------------------------------

    def record_match(
        self,
        query_embedding_id: str,
        matched_embedding_id: str,
        similarity: float,
    ) -> int:
        """Record a re-identification match between two embeddings.

        Returns the match_id of the inserted record.
        """
        ts = _utcnow()
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO reid_matches
                   (query_embedding_id, matched_embedding_id, similarity, timestamp)
                   VALUES (?, ?, ?, ?)""",
                (query_embedding_id, matched_embedding_id, similarity, ts),
            )
            self._conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def get_matches(self, target_id: str) -> list[dict]:
        """Get all cross-camera matches for a target.

        Finds all matches where the target's embeddings appear as either
        the query or matched side, returning the paired target info.

        Returns a list of dicts with ``match_id``, ``query_embedding_id``,
        ``matched_embedding_id``, ``similarity``, ``timestamp``,
        ``query_target_id``, ``query_camera``, ``matched_target_id``,
        ``matched_camera``.
        """
        rows = self._conn.execute(
            """SELECT m.match_id, m.query_embedding_id, m.matched_embedding_id,
                      m.similarity, m.timestamp,
                      eq.target_id AS query_target_id,
                      eq.source_camera AS query_camera,
                      em.target_id AS matched_target_id,
                      em.source_camera AS matched_camera
               FROM reid_matches m
               JOIN reid_embeddings eq ON m.query_embedding_id = eq.embedding_id
               JOIN reid_embeddings em ON m.matched_embedding_id = em.embedding_id
               WHERE eq.target_id = ? OR em.target_id = ?
               ORDER BY m.similarity DESC""",
            (target_id, target_id),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def count_embeddings(self) -> int:
        """Return the total number of stored embeddings."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM reid_embeddings"
        ).fetchone()
        return row["c"]

    def count_matches(self) -> int:
        """Return the total number of recorded matches."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM reid_matches"
        ).fetchone()
        return row["c"]

    def delete_target(self, target_id: str) -> int:
        """Delete all embeddings (and associated matches) for a target.

        Returns the number of embeddings deleted.
        """
        with self._lock:
            # Get embedding IDs for this target
            ids = [
                r["embedding_id"]
                for r in self._conn.execute(
                    "SELECT embedding_id FROM reid_embeddings WHERE target_id = ?",
                    (target_id,),
                ).fetchall()
            ]
            if not ids:
                return 0

            placeholders = ",".join("?" * len(ids))
            self._conn.execute(
                f"DELETE FROM reid_matches WHERE query_embedding_id IN ({placeholders})"
                f" OR matched_embedding_id IN ({placeholders})",
                ids + ids,
            )
            cur = self._conn.execute(
                "DELETE FROM reid_embeddings WHERE target_id = ?",
                (target_id,),
            )
            self._conn.commit()
            return cur.rowcount

    def prune_old_embeddings(self, days: int = 30) -> int:
        """Delete embeddings older than *days* days.  Returns count deleted."""
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._lock:
            # Delete associated matches first
            self._conn.execute(
                """DELETE FROM reid_matches
                   WHERE query_embedding_id IN (
                       SELECT embedding_id FROM reid_embeddings WHERE timestamp < ?
                   ) OR matched_embedding_id IN (
                       SELECT embedding_id FROM reid_embeddings WHERE timestamp < ?
                   )""",
                (cutoff, cutoff),
            )
            cur = self._conn.execute(
                "DELETE FROM reid_embeddings WHERE timestamp < ?", (cutoff,)
            )
            self._conn.commit()
            return cur.rowcount
