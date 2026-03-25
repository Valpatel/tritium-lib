# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""DwellTracker — monitors targets for stationary loitering behavior.

Runs as a periodic check (every 10s) over all tracked targets. When a target
stays within DWELL_RADIUS_M of the same position for longer than
DWELL_THRESHOLD_S, it generates a DwellEvent. Events are broadcast via
EventBus and WebSocket for map visualization (concentric rings).
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid

from tritium_lib.models.dwell import (
    DwellEvent,
    DwellSeverity,
    DwellState,
    DWELL_RADIUS_M,
    DWELL_THRESHOLD_S,
    classify_dwell_severity,
)

logger = logging.getLogger("tactical.dwell")

# Check interval in seconds
_CHECK_INTERVAL_S = 10.0


class DwellTracker:
    """Tracks target dwell times and emits events for loitering detection.

    Args:
        event_bus: Event bus with a .publish(topic, data) method.
        target_tracker: TargetTracker instance to read targets from.
        threshold_s: Minimum dwell time to trigger an event.
        radius_m: Maximum displacement to consider a target dwelling.
    """

    def __init__(
        self,
        event_bus,
        target_tracker,
        threshold_s: float = DWELL_THRESHOLD_S,
        radius_m: float = DWELL_RADIUS_M,
    ) -> None:
        self._event_bus = event_bus
        self._tracker = target_tracker
        self._threshold_s = threshold_s
        self._radius_m = radius_m
        self._lock = threading.Lock()

        self._tracking: dict[str, dict] = {}
        self._active_dwells: dict[str, DwellEvent] = {}
        self._history: list[DwellEvent] = []

        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def active_dwells(self) -> list[DwellEvent]:
        """Return all currently active dwell events."""
        with self._lock:
            return list(self._active_dwells.values())

    @property
    def history(self) -> list[DwellEvent]:
        """Return historical dwell events."""
        with self._lock:
            return list(self._history)

    def start(self) -> None:
        """Start the dwell tracking loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="dwell-tracker")
        self._thread.start()
        logger.info("DwellTracker started (threshold=%ds, radius=%.1fm)", self._threshold_s, self._radius_m)

    def stop(self) -> None:
        """Stop the dwell tracking loop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("DwellTracker stopped")

    def _loop(self) -> None:
        """Main check loop."""
        while self._running:
            try:
                self._check_all_targets()
            except Exception as e:
                logger.debug("DwellTracker check error: %s", e)
            time.sleep(_CHECK_INTERVAL_S)

    def _check_all_targets(self) -> None:
        """Check all tracked targets for dwell behavior."""
        targets = self._tracker.get_all()
        now = time.time()
        seen_ids = set()

        for t in targets:
            seen_ids.add(t.target_id)
            x, y = t.position
            self._check_target(t.target_id, x, y, now, t.name, t.alliance, t.asset_type)

        with self._lock:
            gone_ids = set(self._tracking.keys()) - seen_ids
            for tid in gone_ids:
                if tid in self._active_dwells:
                    self._end_dwell(tid, now)
                self._tracking.pop(tid, None)

    def _check_target(
        self, target_id: str, x: float, y: float, now: float,
        name: str, alliance: str, asset_type: str,
    ) -> None:
        """Check a single target for dwell behavior."""
        with self._lock:
            state = self._tracking.get(target_id)

            if state is None:
                self._tracking[target_id] = {
                    "anchor_x": x,
                    "anchor_y": y,
                    "anchor_time": now,
                    "notified": False,
                }
                return

            dx = x - state["anchor_x"]
            dy = y - state["anchor_y"]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > self._radius_m:
                if target_id in self._active_dwells:
                    self._end_dwell(target_id, now)
                self._tracking[target_id] = {
                    "anchor_x": x,
                    "anchor_y": y,
                    "anchor_time": now,
                    "notified": False,
                }
                return

            dwell_duration = now - state["anchor_time"]

            if dwell_duration >= self._threshold_s:
                if target_id in self._active_dwells:
                    dwell = self._active_dwells[target_id]
                    dwell.duration_s = dwell_duration
                    dwell.severity = classify_dwell_severity(dwell_duration)
                    self._event_bus.publish("dwell_update", dwell.model_dump())
                elif not state.get("notified"):
                    event_id = f"dwell_{uuid.uuid4().hex[:8]}"
                    from datetime import datetime, timezone
                    dwell = DwellEvent(
                        target_id=target_id,
                        event_id=event_id,
                        position_x=state["anchor_x"],
                        position_y=state["anchor_y"],
                        start_time=datetime.fromtimestamp(state["anchor_time"], tz=timezone.utc),
                        duration_s=dwell_duration,
                        state=DwellState.ACTIVE,
                        severity=classify_dwell_severity(dwell_duration),
                        radius_m=self._radius_m,
                        target_name=name,
                        target_alliance=alliance,
                        target_type=asset_type,
                    )
                    self._active_dwells[target_id] = dwell
                    state["notified"] = True
                    logger.info("Dwell detected: %s at (%.1f, %.1f) for %.0fs", target_id, x, y, dwell_duration)
                    self._event_bus.publish("dwell_start", dwell.model_dump())

    def _end_dwell(self, target_id: str, now: float) -> None:
        """End an active dwell event and move to history."""
        dwell = self._active_dwells.pop(target_id, None)
        if dwell is None:
            return
        from datetime import datetime, timezone
        dwell.state = DwellState.ENDED
        dwell.end_time = datetime.fromtimestamp(now, tz=timezone.utc)
        dwell.duration_s = (now - dwell.start_time.timestamp()) if dwell.start_time else 0.0
        self._history.append(dwell)
        if len(self._history) > 200:
            self._history = self._history[-200:]
        logger.info("Dwell ended: %s after %.0fs", target_id, dwell.duration_s)
        self._event_bus.publish("dwell_end", dwell.model_dump())

    def get_dwell_for_target(self, target_id: str) -> DwellEvent | None:
        """Get the active dwell event for a specific target, if any."""
        with self._lock:
            return self._active_dwells.get(target_id)
