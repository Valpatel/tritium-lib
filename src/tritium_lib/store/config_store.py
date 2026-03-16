# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""ConfigStore — persistent key-value system configuration store.

Stores system-level settings like default map position, demo mode preferences,
notification preferences, and plugin configuration. Uses namespaces to avoid
key collisions between different subsystems.

Usage::

    store = ConfigStore("data/config.db")
    store.set("map", "center_lat", "39.7392")
    store.set("map", "center_lng", "-104.9903")
    lat = store.get("map", "center_lat", default="0.0")
    all_map = store.get_namespace("map")  # {"center_lat": "39.7392", ...}
    store.delete("map", "center_lng")
    store.clear_namespace("map")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .base import BaseStore


class ConfigStore(BaseStore):
    """Thread-safe SQLite store for system configuration key-value pairs.

    Values are stored as strings internally. Use :meth:`get_typed` for
    automatic JSON deserialization of complex values.
    """

    _SCHEMAS = [
        """
        CREATE TABLE IF NOT EXISTS config (
            namespace TEXT NOT NULL,
            key       TEXT NOT NULL,
            value     TEXT,
            updated_at REAL NOT NULL,
            PRIMARY KEY (namespace, key)
        );
        """,
        "CREATE INDEX IF NOT EXISTS idx_config_ns ON config(namespace);",
    ]

    def set(self, namespace: str, key: str, value: str) -> None:
        """Set a configuration value.

        Parameters
        ----------
        namespace:
            Logical group (e.g., ``"map"``, ``"demo"``, ``"notifications"``).
        key:
            Setting key within the namespace.
        value:
            String value to store.
        """
        with self._lock:
            self._execute(
                "INSERT OR REPLACE INTO config (namespace, key, value, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (namespace, key, value, time.time()),
            )
            self._commit()

    def get(self, namespace: str, key: str, *, default: str | None = None) -> str | None:
        """Get a configuration value.

        Returns
        -------
        str | None
            The stored value, or *default* if not found.
        """
        with self._lock:
            row = self._fetchone(
                "SELECT value FROM config WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
            return row["value"] if row else default

    def set_json(self, namespace: str, key: str, value: Any) -> None:
        """Set a configuration value by JSON-serializing *value*."""
        self.set(namespace, key, json.dumps(value))

    def get_json(self, namespace: str, key: str, *, default: Any = None) -> Any:
        """Get a configuration value, deserializing from JSON.

        Returns *default* if the key is not found or deserialization fails.
        """
        raw = self.get(namespace, key)
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return default

    def get_namespace(self, namespace: str) -> dict[str, str]:
        """Return all key-value pairs in a namespace as a dict."""
        with self._lock:
            rows = self._fetchall(
                "SELECT key, value FROM config WHERE namespace = ?",
                (namespace,),
            )
            return {row["key"]: row["value"] for row in rows}

    def list_namespaces(self) -> list[str]:
        """Return all distinct namespaces."""
        with self._lock:
            rows = self._fetchall("SELECT DISTINCT namespace FROM config ORDER BY namespace")
            return [row["namespace"] for row in rows]

    def delete(self, namespace: str, key: str) -> bool:
        """Delete a single key. Returns True if the key existed."""
        with self._lock:
            cursor = self._execute(
                "DELETE FROM config WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
            self._commit()
            return cursor.rowcount > 0

    def clear_namespace(self, namespace: str) -> int:
        """Delete all keys in a namespace. Returns the number of keys deleted."""
        with self._lock:
            cursor = self._execute(
                "DELETE FROM config WHERE namespace = ?",
                (namespace,),
            )
            self._commit()
            return cursor.rowcount

    def count(self, namespace: str | None = None) -> int:
        """Count keys, optionally filtered by namespace."""
        with self._lock:
            if namespace is not None:
                row = self._fetchone(
                    "SELECT COUNT(*) as cnt FROM config WHERE namespace = ?",
                    (namespace,),
                )
            else:
                row = self._fetchone("SELECT COUNT(*) as cnt FROM config")
            return row["cnt"] if row else 0

    def set_many(self, namespace: str, items: dict[str, str]) -> None:
        """Set multiple keys in a namespace atomically."""
        now = time.time()
        with self._lock:
            self._executemany(
                "INSERT OR REPLACE INTO config (namespace, key, value, updated_at) "
                "VALUES (?, ?, ?, ?)",
                [(namespace, k, v, now) for k, v in items.items()],
            )
            self._commit()
