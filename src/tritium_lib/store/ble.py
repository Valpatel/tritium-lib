# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SQLite-backed sighting store for BLE and WiFi.

Provides persistent storage for BLE advertisement sightings, WiFi network
sightings, tracked targets, and sensor node positions.  Any Tritium service
(edge server, command center, standalone tool) can instantiate this with a
path to a SQLite database file.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .base import BaseStore

# ---------------------------------------------------------------------------
# SQL schemas
# ---------------------------------------------------------------------------

_SCHEMA_SIGHTINGS = """
CREATE TABLE IF NOT EXISTS ble_sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mac TEXT NOT NULL,
    name TEXT DEFAULT '',
    rssi INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    node_ip TEXT DEFAULT '',
    is_known INTEGER DEFAULT 0,
    seen_count INTEGER DEFAULT 1,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sightings_mac ON ble_sightings(mac);
CREATE INDEX IF NOT EXISTS idx_sightings_ts ON ble_sightings(timestamp);
CREATE INDEX IF NOT EXISTS idx_sightings_node ON ble_sightings(node_id);
"""

_SCHEMA_TARGETS = """
CREATE TABLE IF NOT EXISTS ble_targets (
    mac TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    tracked INTEGER DEFAULT 1,
    color TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
"""

_SCHEMA_WIFI_SIGHTINGS = """
CREATE TABLE IF NOT EXISTS wifi_sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ssid TEXT NOT NULL,
    bssid TEXT NOT NULL,
    rssi INTEGER NOT NULL,
    channel INTEGER DEFAULT 0,
    auth_type TEXT DEFAULT '',
    node_id TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wifi_ssid ON wifi_sightings(ssid);
CREATE INDEX IF NOT EXISTS idx_wifi_bssid ON wifi_sightings(bssid);
CREATE INDEX IF NOT EXISTS idx_wifi_ts ON wifi_sightings(timestamp);
CREATE INDEX IF NOT EXISTS idx_wifi_node ON wifi_sightings(node_id);
"""

_SCHEMA_WIFI_TARGETS = """
CREATE TABLE IF NOT EXISTS wifi_targets (
    bssid TEXT PRIMARY KEY,
    ssid TEXT DEFAULT '',
    label TEXT NOT NULL,
    tracked INTEGER DEFAULT 1,
    color TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
"""

_SCHEMA_NODE_POSITIONS = """
CREATE TABLE IF NOT EXISTS node_positions (
    node_id TEXT PRIMARY KEY,
    x REAL NOT NULL,
    y REAL NOT NULL,
    lat REAL,
    lon REAL,
    label TEXT DEFAULT '',
    updated_at TEXT NOT NULL
);
"""


def _utcnow() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class BleStore(BaseStore):
    """SQLite-backed BLE sighting persistence.

    Shared across the Tritium ecosystem — any service (edge server,
    command center, standalone tool) can instantiate this with a path
    to get persistent BLE tracking.

    Inherits from BaseStore for standardized SQLite WAL setup, thread
    safety, and convenience query methods.
    """

    _SCHEMAS = (
        _SCHEMA_SIGHTINGS,
        _SCHEMA_TARGETS,
        _SCHEMA_WIFI_SIGHTINGS,
        _SCHEMA_WIFI_TARGETS,
        _SCHEMA_NODE_POSITIONS,
    )

    # ------------------------------------------------------------------
    # Sighting recording
    # ------------------------------------------------------------------

    def record_sighting(
        self,
        mac: str,
        name: str,
        rssi: int,
        node_id: str,
        node_ip: str = "",
        is_known: bool = False,
        seen_count: int = 1,
    ) -> int:
        """Record a single BLE sighting.

        Returns the row ID of the inserted record.
        """
        ts = _utcnow()
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO ble_sightings
                   (mac, name, rssi, node_id, node_ip, is_known, seen_count, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (mac, name, rssi, node_id, node_ip, int(is_known), seen_count, ts),
            )
            self._conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def record_sightings_batch(self, sightings: list[dict]) -> int:
        """Batch-insert sightings.

        Each dict should contain: ``mac``, ``name``, ``rssi``,
        ``node_id``, and optionally ``node_ip``, ``is_known``,
        ``seen_count``.

        Returns the number of rows inserted.
        """
        if not sightings:
            return 0
        ts = _utcnow()
        rows = [
            (
                s["mac"],
                s.get("name", ""),
                s["rssi"],
                s["node_id"],
                s.get("node_ip", ""),
                int(s.get("is_known", False)),
                s.get("seen_count", 1),
                s.get("timestamp", ts),
            )
            for s in sightings
        ]
        with self._lock:
            self._conn.executemany(
                """INSERT INTO ble_sightings
                   (mac, name, rssi, node_id, node_ip, is_known, seen_count, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            self._conn.commit()
        return len(rows)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_devices(self, timeout_minutes: int = 2) -> list[dict]:
        """Get unique BLE MACs active in the last *timeout_minutes* minutes.

        Returns a list of dicts suitable for the live BLE tracker UI::

            {
                "mac": str,
                "name": str,
                "is_known": bool,
                "last_rssi": int,
                "strongest_rssi": int,
                "node_count": int,
                "last_seen": str,
                "first_seen": str,
                "total_sightings": int,
                "nodes": [{"node_id": str, "rssi": int, "last_seen": str}, ...],
            }

        The ``nodes`` list contains the strongest RSSI per node, which
        can be fed into
        ``tritium_lib.models.trilateration.estimate_position``.
        """
        cutoff = datetime.now(timezone.utc)
        # Build ISO cutoff string manually to avoid timedelta import
        from datetime import timedelta

        cutoff_str = (cutoff - timedelta(minutes=timeout_minutes)).isoformat()

        # Aggregate per (mac, node_id) to get best RSSI per node
        rows = self._conn.execute(
            """
            SELECT
                mac,
                name,
                MAX(is_known) AS is_known,
                node_id,
                MAX(rssi) AS strongest_rssi,
                MAX(timestamp) AS last_seen,
                MIN(timestamp) AS first_seen,
                SUM(seen_count) AS total_sightings
            FROM ble_sightings
            WHERE timestamp >= ?
            GROUP BY mac, node_id
            ORDER BY mac, strongest_rssi DESC
            """,
            (cutoff_str,),
        ).fetchall()

        # Group by mac
        devices: dict[str, dict] = {}
        for row in rows:
            mac = row["mac"]
            if mac not in devices:
                devices[mac] = {
                    "mac": mac,
                    "name": row["name"],
                    "is_known": bool(row["is_known"]),
                    "last_rssi": row["strongest_rssi"],
                    "strongest_rssi": row["strongest_rssi"],
                    "node_count": 0,
                    "last_seen": row["last_seen"],
                    "first_seen": row["first_seen"],
                    "total_sightings": 0,
                    "nodes": [],
                }
            dev = devices[mac]
            dev["nodes"].append(
                {
                    "node_id": row["node_id"],
                    "rssi": row["strongest_rssi"],
                    "last_seen": row["last_seen"],
                }
            )
            dev["node_count"] = len(dev["nodes"])
            dev["total_sightings"] += row["total_sightings"]
            # Track overall strongest / latest
            if row["strongest_rssi"] > dev["strongest_rssi"]:
                dev["strongest_rssi"] = row["strongest_rssi"]
                dev["last_rssi"] = row["strongest_rssi"]
            if row["last_seen"] > dev["last_seen"]:
                dev["last_seen"] = row["last_seen"]
            if row["first_seen"] < dev["first_seen"]:
                dev["first_seen"] = row["first_seen"]
            # Use latest name if non-empty
            if row["name"]:
                dev["name"] = row["name"]

        return list(devices.values())

    def get_device_history(self, mac: str, limit: int = 200) -> list[dict]:
        """Get sighting history for a specific MAC address."""
        rows = self._conn.execute(
            """SELECT id, mac, name, rssi, node_id, node_ip,
                      is_known, seen_count, timestamp
               FROM ble_sightings
               WHERE mac = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (mac, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_device_summary(self) -> dict:
        """Summary stats for the BLE sighting database.

        Returns ``total_unique_macs``, ``active_count`` (last 2 min),
        ``known_count``, ``tracked_count``, ``sighting_count``.
        """
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()

        row = self._conn.execute(
            """SELECT
                COUNT(DISTINCT mac) AS total_unique_macs,
                SUM(seen_count) AS sighting_count
               FROM ble_sightings"""
        ).fetchone()

        active = self._conn.execute(
            "SELECT COUNT(DISTINCT mac) AS c FROM ble_sightings WHERE timestamp >= ?",
            (cutoff,),
        ).fetchone()

        known = self._conn.execute(
            "SELECT COUNT(DISTINCT mac) AS c FROM ble_sightings WHERE is_known = 1",
        ).fetchone()

        tracked = self._conn.execute(
            "SELECT COUNT(*) AS c FROM ble_targets WHERE tracked = 1",
        ).fetchone()

        return {
            "total_unique_macs": row["total_unique_macs"] or 0,
            "active_count": active["c"] or 0,
            "known_count": known["c"] or 0,
            "tracked_count": tracked["c"] or 0,
            "sighting_count": row["sighting_count"] or 0,
        }

    # ------------------------------------------------------------------
    # Target tracking
    # ------------------------------------------------------------------

    def add_target(self, mac: str, label: str, color: str = "") -> dict:
        """Mark a BLE MAC as a tracked target with a human label.

        Returns the target record as a dict.
        """
        ts = _utcnow()
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO ble_targets (mac, label, tracked, color, created_at)
                   VALUES (?, ?, 1, ?, ?)""",
                (mac, label, color, ts),
            )
            self._conn.commit()
        return {"mac": mac, "label": label, "tracked": True, "color": color, "created_at": ts}

    def remove_target(self, mac: str) -> bool:
        """Remove a tracked target.  Returns ``True`` if the target existed."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM ble_targets WHERE mac = ?", (mac,))
            self._conn.commit()
            return cur.rowcount > 0

    def list_targets(self) -> list[dict]:
        """List all tracked targets with their latest sighting info."""
        rows = self._conn.execute(
            """SELECT t.mac, t.label, t.tracked, t.color, t.created_at,
                      s.rssi AS last_rssi, s.node_id AS last_node,
                      s.timestamp AS last_seen
               FROM ble_targets t
               LEFT JOIN (
                   SELECT mac, rssi, node_id, timestamp,
                          ROW_NUMBER() OVER (PARTITION BY mac ORDER BY timestamp DESC) AS rn
                   FROM ble_sightings
               ) s ON t.mac = s.mac AND s.rn = 1
               ORDER BY t.label"""
        ).fetchall()
        return [dict(r) for r in rows]

    def update_target(
        self,
        mac: str,
        label: str | None = None,
        color: str | None = None,
        tracked: bool | None = None,
    ) -> bool:
        """Update a tracked target's properties.

        Only the provided (non-``None``) fields are modified.  Returns
        ``True`` if the target existed and was updated.
        """
        updates: list[str] = []
        params: list[object] = []
        if label is not None:
            updates.append("label = ?")
            params.append(label)
        if color is not None:
            updates.append("color = ?")
            params.append(color)
        if tracked is not None:
            updates.append("tracked = ?")
            params.append(int(tracked))
        if not updates:
            return False
        params.append(mac)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE ble_targets SET {', '.join(updates)} WHERE mac = ?",
                params,
            )
            self._conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Node positions
    # ------------------------------------------------------------------

    def set_node_position(
        self,
        node_id: str,
        x: float,
        y: float,
        lat: float | None = None,
        lon: float | None = None,
        label: str = "",
    ) -> None:
        """Set or update a sensor node's position."""
        ts = _utcnow()
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO node_positions
                   (node_id, x, y, lat, lon, label, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (node_id, x, y, lat, lon, label, ts),
            )
            self._conn.commit()

    def get_node_positions(self) -> dict[str, dict]:
        """Get all node positions.

        Returns ``{node_id: {x, y, lat, lon, label, updated_at}}``.
        """
        rows = self._conn.execute("SELECT * FROM node_positions").fetchall()
        return {
            r["node_id"]: {
                "x": r["x"],
                "y": r["y"],
                "lat": r["lat"],
                "lon": r["lon"],
                "label": r["label"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        }

    def get_node_position(self, node_id: str) -> dict | None:
        """Get a single node's position, or ``None`` if not found."""
        row = self._conn.execute(
            "SELECT * FROM node_positions WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "x": row["x"],
            "y": row["y"],
            "lat": row["lat"],
            "lon": row["lon"],
            "label": row["label"],
            "updated_at": row["updated_at"],
        }

    def remove_node_position(self, node_id: str) -> bool:
        """Remove a node position.  Returns ``True`` if it existed."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM node_positions WHERE node_id = ?", (node_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # WiFi sighting recording
    # ------------------------------------------------------------------

    def record_wifi_sighting(
        self,
        ssid: str,
        bssid: str,
        rssi: int,
        channel: int = 0,
        auth_type: str = "",
        node_id: str = "",
        timestamp: str | None = None,
    ) -> int:
        """Record a single WiFi network sighting.  Returns row ID."""
        ts = timestamp or _utcnow()
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO wifi_sightings
                   (ssid, bssid, rssi, channel, auth_type, node_id, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (ssid, bssid, rssi, channel, auth_type, node_id, ts),
            )
            self._conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def record_wifi_sightings_batch(self, sightings: list[dict]) -> int:
        """Batch-insert WiFi sightings.

        Each dict should contain: ``ssid``, ``bssid``, ``rssi``,
        ``channel``, ``auth_type``, ``node_id``.

        Returns the number of rows inserted.
        """
        if not sightings:
            return 0
        ts = _utcnow()
        rows = [
            (
                s["ssid"],
                s["bssid"],
                s["rssi"],
                s.get("channel", 0),
                s.get("auth_type", ""),
                s.get("node_id", ""),
                s.get("timestamp", ts),
            )
            for s in sightings
        ]
        with self._lock:
            self._conn.executemany(
                """INSERT INTO wifi_sightings
                   (ssid, bssid, rssi, channel, auth_type, node_id, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            self._conn.commit()
        return len(rows)

    # ------------------------------------------------------------------
    # WiFi queries
    # ------------------------------------------------------------------

    def get_active_wifi_networks(self, timeout_minutes: int = 5) -> list[dict]:
        """Get unique WiFi networks seen in the last *timeout_minutes* minutes.

        Returns a list of dicts::

            {
                "ssid": str,
                "bssid": str,
                "strongest_rssi": int,
                "channel": int,
                "auth_type": str,
                "node_count": int,
                "last_seen": str,
                "nodes": [{"node_id": str, "rssi": int, "last_seen": str}, ...],
            }
        """
        from datetime import timedelta

        cutoff_str = (
            datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)
        ).isoformat()

        rows = self._conn.execute(
            """
            SELECT
                bssid,
                ssid,
                node_id,
                MAX(rssi) AS strongest_rssi,
                MAX(channel) AS channel,
                MAX(auth_type) AS auth_type,
                MAX(timestamp) AS last_seen
            FROM wifi_sightings
            WHERE timestamp >= ?
            GROUP BY bssid, node_id
            ORDER BY bssid, strongest_rssi DESC
            """,
            (cutoff_str,),
        ).fetchall()

        networks: dict[str, dict] = {}
        for row in rows:
            bssid = row["bssid"]
            if bssid not in networks:
                networks[bssid] = {
                    "ssid": row["ssid"],
                    "bssid": bssid,
                    "strongest_rssi": row["strongest_rssi"],
                    "channel": row["channel"],
                    "auth_type": row["auth_type"],
                    "node_count": 0,
                    "last_seen": row["last_seen"],
                    "nodes": [],
                }
            net = networks[bssid]
            net["nodes"].append(
                {
                    "node_id": row["node_id"],
                    "rssi": row["strongest_rssi"],
                    "last_seen": row["last_seen"],
                }
            )
            net["node_count"] = len(net["nodes"])
            if row["strongest_rssi"] > net["strongest_rssi"]:
                net["strongest_rssi"] = row["strongest_rssi"]
            if row["last_seen"] > net["last_seen"]:
                net["last_seen"] = row["last_seen"]
            if row["ssid"]:
                net["ssid"] = row["ssid"]

        return list(networks.values())

    def get_wifi_history(self, bssid: str, limit: int = 200) -> list[dict]:
        """Sighting history for a specific BSSID."""
        rows = self._conn.execute(
            """SELECT id, ssid, bssid, rssi, channel, auth_type,
                      node_id, timestamp
               FROM wifi_sightings
               WHERE bssid = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (bssid, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_wifi_summary(self) -> dict:
        """Summary stats for the WiFi sighting database.

        Returns ``total_unique_networks``, ``active_count`` (last 5 min),
        ``tracked_count``, ``sighting_count``.
        """
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()

        row = self._conn.execute(
            """SELECT
                COUNT(DISTINCT bssid) AS total_unique_networks,
                COUNT(*) AS sighting_count
               FROM wifi_sightings"""
        ).fetchone()

        active = self._conn.execute(
            "SELECT COUNT(DISTINCT bssid) AS c FROM wifi_sightings WHERE timestamp >= ?",
            (cutoff,),
        ).fetchone()

        tracked = self._conn.execute(
            "SELECT COUNT(*) AS c FROM wifi_targets WHERE tracked = 1",
        ).fetchone()

        return {
            "total_unique_networks": row["total_unique_networks"] or 0,
            "active_count": active["c"] or 0,
            "tracked_count": tracked["c"] or 0,
            "sighting_count": row["sighting_count"] or 0,
        }

    # ------------------------------------------------------------------
    # WiFi target tracking
    # ------------------------------------------------------------------

    def add_wifi_target(self, bssid: str, label: str, ssid: str = "", color: str = "") -> dict:
        """Mark a WiFi BSSID as a tracked target with a human label.

        Returns the target record as a dict.
        """
        ts = _utcnow()
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO wifi_targets
                   (bssid, ssid, label, tracked, color, created_at)
                   VALUES (?, ?, ?, 1, ?, ?)""",
                (bssid, ssid, label, color, ts),
            )
            self._conn.commit()
        return {
            "bssid": bssid,
            "ssid": ssid,
            "label": label,
            "tracked": True,
            "color": color,
            "created_at": ts,
        }

    def remove_wifi_target(self, bssid: str) -> bool:
        """Remove a tracked WiFi target.  Returns ``True`` if it existed."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM wifi_targets WHERE bssid = ?", (bssid,))
            self._conn.commit()
            return cur.rowcount > 0

    def list_wifi_targets(self) -> list[dict]:
        """List all tracked WiFi targets."""
        rows = self._conn.execute(
            """SELECT bssid, ssid, label, tracked, color, created_at
               FROM wifi_targets
               ORDER BY label"""
        ).fetchall()
        return [dict(r) for r in rows]

    def update_wifi_target(
        self,
        bssid: str,
        label: str | None = None,
        ssid: str | None = None,
        color: str | None = None,
        tracked: bool | None = None,
    ) -> bool:
        """Update a tracked WiFi target's properties.

        Only the provided (non-``None``) fields are modified.  Returns
        ``True`` if the target existed and was updated.
        """
        updates: list[str] = []
        params: list[object] = []
        if label is not None:
            updates.append("label = ?")
            params.append(label)
        if ssid is not None:
            updates.append("ssid = ?")
            params.append(ssid)
        if color is not None:
            updates.append("color = ?")
            params.append(color)
        if tracked is not None:
            updates.append("tracked = ?")
            params.append(int(tracked))
        if not updates:
            return False
        params.append(bssid)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE wifi_targets SET {', '.join(updates)} WHERE bssid = ?",
                params,
            )
            self._conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # WiFi maintenance
    # ------------------------------------------------------------------

    def prune_old_wifi_sightings(self, days: int = 7) -> int:
        """Delete WiFi sightings older than *days* days.  Returns count deleted."""
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM wifi_sightings WHERE timestamp < ?", (cutoff,)
            )
            self._conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune_old_sightings(self, days: int = 7) -> int:
        """Delete sightings older than *days* days.  Returns count deleted."""
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM ble_sightings WHERE timestamp < ?", (cutoff,)
            )
            self._conn.commit()
            return cur.rowcount

    def get_stats(self) -> dict:
        """Database statistics.

        Returns ``db_size_bytes``, ``sighting_count``, ``device_count``,
        ``target_count``, ``node_count``, ``oldest_sighting``,
        ``newest_sighting``.
        """
        sighting_row = self._conn.execute(
            """SELECT COUNT(*) AS cnt,
                      MIN(timestamp) AS oldest,
                      MAX(timestamp) AS newest
               FROM ble_sightings"""
        ).fetchone()

        device_count = self._conn.execute(
            "SELECT COUNT(DISTINCT mac) AS c FROM ble_sightings"
        ).fetchone()["c"]

        target_count = self._conn.execute(
            "SELECT COUNT(*) AS c FROM ble_targets"
        ).fetchone()["c"]

        node_count = self._conn.execute(
            "SELECT COUNT(*) AS c FROM node_positions"
        ).fetchone()["c"]

        # File size (0 for in-memory databases)
        try:
            db_size = os.path.getsize(self._db_path) if self._db_path != ":memory:" else 0
        except OSError:
            db_size = 0

        wifi_sighting_count = self._conn.execute(
            "SELECT COUNT(*) AS c FROM wifi_sightings"
        ).fetchone()["c"]

        wifi_network_count = self._conn.execute(
            "SELECT COUNT(DISTINCT bssid) AS c FROM wifi_sightings"
        ).fetchone()["c"]

        return {
            "db_size_bytes": db_size,
            "sighting_count": sighting_row["cnt"],
            "device_count": device_count,
            "target_count": target_count,
            "node_count": node_count,
            "oldest_sighting": sighting_row["oldest"],
            "newest_sighting": sighting_row["newest"],
            "wifi_sighting_count": wifi_sighting_count,
            "wifi_network_count": wifi_network_count,
        }
