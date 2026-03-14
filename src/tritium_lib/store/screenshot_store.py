# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SQLite-backed screenshot store for tactical map captures.

Persists PNG screenshots captured from the tactical map with metadata
including timestamp, operator name, description, and the raw PNG binary.
Any Tritium service (command center, edge fleet server) can instantiate
this with a path to a SQLite database file.
"""

from __future__ import annotations

import time
import uuid

from .base import BaseStore

# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_SCHEMA_SCREENSHOTS = """
CREATE TABLE IF NOT EXISTS screenshots (
    screenshot_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    operator TEXT NOT NULL DEFAULT 'unknown',
    description TEXT NOT NULL DEFAULT '',
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0,
    file_size INTEGER NOT NULL DEFAULT 0,
    png_data BLOB NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_screenshots_timestamp ON screenshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_screenshots_operator ON screenshots(operator);
"""


class ScreenshotStore(BaseStore):
    """Thread-safe store for tactical map screenshots.

    Each screenshot is stored with:
    - screenshot_id: unique UUID
    - timestamp: Unix epoch when captured
    - operator: who took the screenshot
    - description: optional text label
    - width/height: image dimensions in pixels
    - file_size: PNG byte count
    - png_data: raw PNG binary
    - tags: JSON array of string tags
    """

    _SCHEMAS = [_SCHEMA_SCREENSHOTS]

    def save(
        self,
        png_data: bytes,
        *,
        operator: str = "unknown",
        description: str = "",
        width: int = 0,
        height: int = 0,
        tags: list[str] | None = None,
    ) -> dict:
        """Save a screenshot and return its metadata (without binary).

        Parameters
        ----------
        png_data:
            Raw PNG image bytes.
        operator:
            Name or ID of the operator who captured it.
        description:
            Optional text description.
        width, height:
            Image dimensions in pixels.
        tags:
            Optional list of string tags.

        Returns
        -------
        dict with screenshot metadata (no png_data).
        """
        import json

        sid = str(uuid.uuid4())
        now = time.time()
        tags_json = json.dumps(tags or [])

        with self._lock:
            self._execute(
                """INSERT INTO screenshots
                   (screenshot_id, timestamp, operator, description,
                    width, height, file_size, png_data, tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sid, now, operator, description,
                 width, height, len(png_data), png_data, tags_json),
            )
            self._commit()

        return {
            "screenshot_id": sid,
            "timestamp": now,
            "operator": operator,
            "description": description,
            "width": width,
            "height": height,
            "file_size": len(png_data),
            "tags": tags or [],
        }

    def list_screenshots(
        self, *, limit: int = 50, offset: int = 0, operator: str | None = None
    ) -> list[dict]:
        """List screenshot metadata (without binary data).

        Parameters
        ----------
        limit:
            Maximum number of results.
        offset:
            Skip this many results (for pagination).
        operator:
            Filter by operator name (optional).

        Returns
        -------
        List of metadata dicts, newest first.
        """
        import json

        with self._lock:
            if operator:
                rows = self._fetchall(
                    """SELECT screenshot_id, timestamp, operator, description,
                              width, height, file_size, tags
                       FROM screenshots
                       WHERE operator = ?
                       ORDER BY timestamp DESC
                       LIMIT ? OFFSET ?""",
                    (operator, limit, offset),
                )
            else:
                rows = self._fetchall(
                    """SELECT screenshot_id, timestamp, operator, description,
                              width, height, file_size, tags
                       FROM screenshots
                       ORDER BY timestamp DESC
                       LIMIT ? OFFSET ?""",
                    (limit, offset),
                )

        return [
            {
                "screenshot_id": row["screenshot_id"],
                "timestamp": row["timestamp"],
                "operator": row["operator"],
                "description": row["description"],
                "width": row["width"],
                "height": row["height"],
                "file_size": row["file_size"],
                "tags": json.loads(row["tags"]),
            }
            for row in rows
        ]

    def get(self, screenshot_id: str) -> dict | None:
        """Get a screenshot including its PNG binary.

        Returns
        -------
        dict with all fields including 'png_data', or None if not found.
        """
        import json

        with self._lock:
            row = self._fetchone(
                "SELECT * FROM screenshots WHERE screenshot_id = ?",
                (screenshot_id,),
            )

        if row is None:
            return None

        return {
            "screenshot_id": row["screenshot_id"],
            "timestamp": row["timestamp"],
            "operator": row["operator"],
            "description": row["description"],
            "width": row["width"],
            "height": row["height"],
            "file_size": row["file_size"],
            "png_data": bytes(row["png_data"]),
            "tags": json.loads(row["tags"]),
        }

    def delete(self, screenshot_id: str) -> bool:
        """Delete a screenshot by ID.

        Returns True if a screenshot was deleted, False if not found.
        """
        with self._lock:
            cursor = self._execute(
                "DELETE FROM screenshots WHERE screenshot_id = ?",
                (screenshot_id,),
            )
            self._commit()
            return cursor.rowcount > 0

    def count(self) -> int:
        """Return total number of stored screenshots."""
        with self._lock:
            row = self._fetchone("SELECT COUNT(*) as cnt FROM screenshots")
            return row["cnt"] if row else 0
