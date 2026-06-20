# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""EmbodimentRegistry — the shared registry of embodiment slots + occupants.

An *embodiment* is one controllable body in the Tritium world: a simulated
unit, a real MQTT robot, a sensor node, a PTZ camera. Each slot starts driven
by an in-repo **stand-in** AI; a **Graphling** may later *check in* to take the
shift and drive it through the public ``/api/embodiments/*`` SDK, then *check
out* and go home with its memories.

This module owns ONLY the state and the pure state operations. It deliberately
lives in ``tritium_lib.store`` — the lowest shared layer — so BOTH the SC HTTP
router (``app.routers.embodiments``) AND the simulation engine / MQTT bridge
import the SAME singleton from here. Previously the engine imported the registry
from ``app.routers.embodiments``, a layering inversion (a lower layer reaching
up into the web router); routing everything through this store fixes that and
gives the registry a natural home for persistence.

Persistence (opt-in via :func:`configure_persistence`) is a lightweight JSON
file — no SQLite, no framework deps (tritium-lib rule #3). It carries the
durable facts across a Tritium restart:

  * the **slot inventory** (what bodies exist), re-loaded as *stand-in* — a real
    Graphling is responsible for re-checking-in after a reboot, so occupancy and
    transient perception/pending-action are NOT restored; and
  * the **per-Graphling leaderboard stats** (kills/score/shifts), which are
    cumulative for the Graphling across shifts and SHOULD survive a restart.

HTTP concerns (rate limiting, FastAPI request models, WebSocket broadcast) stay
in the SC router; they have no place in the shared library.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import RLock
from typing import Any


def _now() -> float:
    return time.time()


def _fresh_stats(ts: float | None = None) -> dict[str, Any]:
    return {
        "kills": 0, "score": 0, "shifts_completed": 0,
        "total_shift_seconds": 0, "last_active_ts": ts if ts is not None else _now(),
        "kind": None, "label": None,
    }


class EmbodimentRegistry:
    """Thread-safe registry of embodiment slots and per-Graphling stats.

    All mutating operations take the registry lock. The ``embodiments`` and
    ``stats`` dicts are exposed as attributes so the SC router can operate on the
    SAME objects under the SAME lock — the router holds the HTTP surface, this
    class holds the state.
    """

    def __init__(self) -> None:
        self.lock = RLock()
        self.embodiments: dict[str, dict[str, Any]] = {}
        self.stats: dict[str, dict[str, Any]] = {}
        self._persist_path: Path | None = None

    # ------------------------------------------------------------------
    # Persistence (opt-in; no-op until configured)
    # ------------------------------------------------------------------
    def configure_persistence(self, path: str | Path | None) -> None:
        """Enable JSON persistence at ``path`` and load any existing state.

        Idempotent. Passing ``None`` disables persistence (used by tests). On
        load, slots come back as *stand-in* (occupancy is never restored) and
        leaderboard stats come back verbatim.
        """
        if path is None:
            self._persist_path = None
            return
        self._persist_path = Path(path)
        self._load()

    def _load(self) -> None:
        p = self._persist_path
        if p is None or not p.exists():
            return
        try:
            data = json.loads(p.read_text())
        except Exception:
            return
        with self.lock:
            for rec in data.get("slots", []):
                eid = rec.get("embodiment_id")
                if not eid:
                    continue
                # Restore the slot as a stand-in — a Graphling must re-checkin
                # after a reboot; transient occupancy/perception do not survive.
                self.embodiments[eid] = {
                    "embodiment_id": eid,
                    "kind": rec.get("kind", "stand-in"),
                    "label": rec.get("label") or eid,
                    "capabilities": list(rec.get("capabilities") or []),
                    "occupant": "stand-in",
                    "graphling_id": None,
                    "checkin_ts": None,
                    "last_checkout_ts": rec.get("last_checkout_ts"),
                    "registered_ts": rec.get("registered_ts") or _now(),
                    "pending_action": None,
                    "perception": None,
                }
            for gid, s in (data.get("stats") or {}).items():
                merged = _fresh_stats()
                merged.update({k: v for k, v in s.items() if k in merged})
                self.stats[gid] = merged

    def save(self) -> None:
        """Persist the durable registry snapshot (best-effort, never raises)."""
        p = self._persist_path
        if p is None:
            return
        try:
            with self.lock:
                slots = [
                    {
                        "embodiment_id": rec["embodiment_id"],
                        "kind": rec.get("kind"),
                        "label": rec.get("label"),
                        "capabilities": list(rec.get("capabilities") or []),
                        "registered_ts": rec.get("registered_ts"),
                        "last_checkout_ts": rec.get("last_checkout_ts"),
                    }
                    for rec in self.embodiments.values()
                ]
                payload = {"slots": slots, "stats": dict(self.stats)}
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(p)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Slot lifecycle + perception/action plumbing
    # ------------------------------------------------------------------
    def register(
        self,
        embodiment_id: str,
        *,
        kind: str = "stand-in",
        label: str | None = None,
        capabilities: list[str] | None = None,
    ) -> dict[str, Any]:
        """Idempotently register an embodiment slot (returns the record)."""
        with self.lock:
            existing = self.embodiments.get(embodiment_id)
            if existing:
                return existing
            rec = {
                "embodiment_id": embodiment_id,
                "kind": kind,
                "label": label or embodiment_id,
                "capabilities": list(capabilities or []),
                "occupant": "stand-in",     # "stand-in" | "graphling"
                "graphling_id": None,
                "checkin_ts": None,
                "last_checkout_ts": None,
                "registered_ts": _now(),
                "pending_action": None,     # occupant's last decided action (consumed by the engine)
                "perception": None,         # engine-published egocentric view (delivered to the occupant)
            }
            self.embodiments[embodiment_id] = rec
            return rec

    def is_occupied(self, embodiment_id: str) -> bool:
        """True when a Graphling is on shift in this slot (stand-in suppressed)."""
        with self.lock:
            rec = self.embodiments.get(embodiment_id)
            return bool(rec and rec.get("occupant") == "graphling")

    def pop_pending_action(self, embodiment_id: str) -> dict[str, Any] | None:
        """Consume the occupant's pending action (engine calls this each tick)."""
        with self.lock:
            rec = self.embodiments.get(embodiment_id)
            if not rec or rec.get("occupant") != "graphling":
                return None
            act = rec.get("pending_action")
            rec["pending_action"] = None
            return act

    def occupied_ids(self) -> list[str]:
        """IDs of every embodiment currently on shift (occupant == graphling)."""
        with self.lock:
            return [eid for eid, r in self.embodiments.items()
                    if r.get("occupant") == "graphling"]

    def set_perception(self, embodiment_id: str, snapshot: dict[str, Any] | None) -> None:
        """Engine pushes the occupied unit's egocentric perception each tick."""
        with self.lock:
            rec = self.embodiments.get(embodiment_id)
            if rec is not None:
                rec["perception"] = snapshot

    def get_perception(self, embodiment_id: str) -> dict[str, Any] | None:
        """Latest perception snapshot for a slot (None if unknown / none published)."""
        with self.lock:
            rec = self.embodiments.get(embodiment_id)
            return rec.get("perception") if rec else None

    def deregister_silent(self, embodiment_id: str) -> bool:
        """Remove a slot without raising if it's already gone (engine teardown)."""
        with self.lock:
            removed = self.embodiments.pop(embodiment_id, None) is not None
        if removed:
            self.save()
        return removed

    # ------------------------------------------------------------------
    # Per-Graphling leaderboard stats
    # ------------------------------------------------------------------
    def record_kill(self, graphling_id: str, points: int = 100) -> None:
        """A Graphling killed a hostile: +1 kill, +points."""
        if not graphling_id:
            return
        with self.lock:
            s = self.stats.setdefault(graphling_id, _fresh_stats())
            s["kills"] += 1
            s["score"] += int(points)
            s["last_active_ts"] = _now()
        self.save()

    def record_score(self, graphling_id: str, points: int) -> None:
        """Arbitrary score event for a Graphling."""
        if not graphling_id:
            return
        with self.lock:
            s = self.stats.setdefault(graphling_id, _fresh_stats())
            s["score"] += int(points)
            s["last_active_ts"] = _now()
        self.save()


# Process-wide singleton. Both the SC router and the simulation engine import
# from here, so they share ONE registry instance.
REGISTRY = EmbodimentRegistry()

# Shared mutable state, exposed for the SC router (which operates on the same
# objects under the same lock) and for legacy importers.
_embodiments = REGISTRY.embodiments
_lock = REGISTRY.lock
_graphling_stats = REGISTRY.stats


# ----------------------------------------------------------------------
# Module-level free functions — the stable API that the SC router re-exports
# and that the engine / MQTT bridge import directly (fixing the old inversion).
# ----------------------------------------------------------------------
def configure_persistence(path: str | Path | None) -> None:
    REGISTRY.configure_persistence(path)


def register_embodiment(
    embodiment_id: str,
    *,
    kind: str = "stand-in",
    label: str | None = None,
    capabilities: list[str] | None = None,
) -> dict[str, Any]:
    return REGISTRY.register(
        embodiment_id, kind=kind, label=label, capabilities=capabilities
    )


def is_occupied(embodiment_id: str) -> bool:
    return REGISTRY.is_occupied(embodiment_id)


def pop_pending_action(embodiment_id: str) -> dict[str, Any] | None:
    return REGISTRY.pop_pending_action(embodiment_id)


def occupied_ids() -> list[str]:
    return REGISTRY.occupied_ids()


def set_perception(embodiment_id: str, snapshot: dict[str, Any] | None) -> None:
    REGISTRY.set_perception(embodiment_id, snapshot)


def get_perception(embodiment_id: str) -> dict[str, Any] | None:
    return REGISTRY.get_perception(embodiment_id)


def deregister_embodiment_silent(embodiment_id: str) -> bool:
    return REGISTRY.deregister_silent(embodiment_id)


def record_graphling_kill(graphling_id: str, points: int = 100) -> None:
    REGISTRY.record_kill(graphling_id, points=points)


def record_graphling_score(graphling_id: str, points: int) -> None:
    REGISTRY.record_score(graphling_id, points)


def save() -> None:
    REGISTRY.save()


__all__ = [
    "EmbodimentRegistry", "REGISTRY",
    "configure_persistence", "save",
    "register_embodiment", "is_occupied", "pop_pending_action",
    "occupied_ids", "set_perception", "get_perception",
    "deregister_embodiment_silent",
    "record_graphling_kill", "record_graphling_score",
    "_embodiments", "_lock", "_graphling_stats", "_now",
]
