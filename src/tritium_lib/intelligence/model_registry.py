# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""ModelRegistry — versioned ML model storage for federation sharing.

SQLite-backed registry for saving, loading, and listing trained ML models.
Used by tritium-sc to store correlation/classification models and share
them across federation sites via the model export/import API.

Usage::

    registry = ModelRegistry("data/model_registry.db")
    registry.save_model("correlation", "1.0.0", model_bytes, {"accuracy": 0.95})
    model = registry.load_model("correlation", "1.0.0")
    latest = registry.get_latest("correlation")
    all_models = registry.list_models()
"""
from __future__ import annotations

import time
from typing import Any, Optional

from tritium_lib.store.base import BaseStore


class ModelRegistry(BaseStore):
    """Versioned ML model storage backed by SQLite.

    Stores serialized model bytes alongside metadata (accuracy, training
    count, feature names, etc.). Supports versioning so multiple iterations
    of the same model can coexist.
    """

    _SCHEMAS = (
        """
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            data BLOB NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            UNIQUE(name, version)
        );
        CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
        CREATE INDEX IF NOT EXISTS idx_models_created ON models(created_at);
        """,
    )

    def save_model(
        self,
        name: str,
        version: str,
        data: bytes,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Save a model to the registry.

        Args:
            name: Model name (e.g. "correlation", "ble_classifier").
            version: Semantic version string (e.g. "1.0.0").
            data: Serialized model bytes (pickle, TFLite, ONNX, etc.).
            metadata: Optional metadata dict (accuracy, training_count, etc.).

        Returns:
            Dict with id, name, version, size_bytes, created_at.

        Raises:
            ValueError: If name or version is empty, or data is empty.
        """
        import json

        if not name or not name.strip():
            raise ValueError("Model name cannot be empty")
        if not version or not version.strip():
            raise ValueError("Model version cannot be empty")
        if not data:
            raise ValueError("Model data cannot be empty")

        name = name.strip()
        version = version.strip()
        meta_json = json.dumps(metadata or {})
        created_at = time.time()
        size_bytes = len(data)

        with self._lock:
            self._execute(
                """
                INSERT OR REPLACE INTO models
                    (name, version, data, metadata, created_at, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, version, data, meta_json, created_at, size_bytes),
            )
            self._commit()

            row = self._fetchone(
                "SELECT id FROM models WHERE name = ? AND version = ?",
                (name, version),
            )

        return {
            "id": row["id"] if row else None,
            "name": name,
            "version": version,
            "size_bytes": size_bytes,
            "created_at": created_at,
        }

    def load_model(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Load a model from the registry.

        Args:
            name: Model name.
            version: Specific version to load. If None, loads the latest.

        Returns:
            Dict with name, version, data (bytes), metadata, created_at,
            size_bytes. Returns None if not found.
        """
        import json

        with self._lock:
            if version:
                row = self._fetchone(
                    "SELECT * FROM models WHERE name = ? AND version = ?",
                    (name, version),
                )
            else:
                row = self._fetchone(
                    "SELECT * FROM models WHERE name = ? ORDER BY created_at DESC LIMIT 1",
                    (name,),
                )

        if row is None:
            return None

        try:
            metadata = json.loads(row["metadata"])
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        return {
            "id": row["id"],
            "name": row["name"],
            "version": row["version"],
            "data": bytes(row["data"]),
            "metadata": metadata,
            "created_at": row["created_at"],
            "size_bytes": row["size_bytes"],
        }

    def get_latest(self, name: str) -> Optional[dict[str, Any]]:
        """Get the latest version of a named model.

        Convenience wrapper around load_model(name, version=None).
        """
        return self.load_model(name, version=None)

    def list_models(
        self,
        name: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List models in the registry (without data blobs).

        Args:
            name: Optional filter by model name.
            limit: Max number of results.

        Returns:
            List of dicts with name, version, metadata, created_at, size_bytes.
        """
        import json

        with self._lock:
            if name:
                rows = self._fetchall(
                    """
                    SELECT id, name, version, metadata, created_at, size_bytes
                    FROM models WHERE name = ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (name, limit),
                )
            else:
                rows = self._fetchall(
                    """
                    SELECT id, name, version, metadata, created_at, size_bytes
                    FROM models ORDER BY created_at DESC LIMIT ?
                    """,
                    (limit,),
                )

        result = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            result.append({
                "id": row["id"],
                "name": row["name"],
                "version": row["version"],
                "metadata": metadata,
                "created_at": row["created_at"],
                "size_bytes": row["size_bytes"],
            })
        return result

    def delete_model(self, name: str, version: str) -> bool:
        """Delete a specific model version.

        Returns:
            True if a model was deleted, False if not found.
        """
        with self._lock:
            cursor = self._execute(
                "DELETE FROM models WHERE name = ? AND version = ?",
                (name, version),
            )
            self._commit()
            return cursor.rowcount > 0

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dict with total_models, unique_names, total_size_bytes.
        """
        with self._lock:
            row = self._fetchone(
                """
                SELECT
                    COUNT(*) as total_models,
                    COUNT(DISTINCT name) as unique_names,
                    COALESCE(SUM(size_bytes), 0) as total_size_bytes
                FROM models
                """
            )

        if row is None:
            return {"total_models": 0, "unique_names": 0, "total_size_bytes": 0}

        return {
            "total_models": row["total_models"],
            "unique_names": row["unique_names"],
            "total_size_bytes": row["total_size_bytes"],
        }
