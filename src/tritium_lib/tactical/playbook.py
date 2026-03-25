# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tactical playbook system — predefined response patterns for common scenarios.

A Playbook is a named sequence of PlaybookActions that encode a tactical
response pattern. The PlaybookRunner executes a playbook against the live
tracking pipeline (TargetTracker, GeofenceEngine, AlertEngine, etc.),
stepping through actions and recording results.

Built-in playbooks:
    unknown_entry — new target enters area -> classify -> monitor -> alert if threat
    pursuit       — target leaves zone -> track trajectory -> predict destination
    gathering     — multiple targets converge -> monitor density -> alert if threshold
    sweep         — systematic search of an area -> zone-by-zone -> report findings
    perimeter     — establish geofence -> monitor all crossings -> log everything

Quick start::

    from tritium_lib.tactical.playbook import (
        PlaybookRunner, BUILTIN_PLAYBOOKS, load_builtin_playbooks,
    )
    from tritium_lib.tracking import TargetTracker, GeofenceEngine
    from tritium_lib.events.bus import EventBus

    bus = EventBus()
    tracker = TargetTracker(event_bus=bus)
    geofence = GeofenceEngine(event_bus=bus)

    runner = PlaybookRunner(tracker=tracker, geofence=geofence, event_bus=bus)
    for pb in load_builtin_playbooks():
        runner.register_playbook(pb)

    # Execute a playbook against a specific target
    result = runner.execute("unknown_entry", context={"target_id": "ble_aabbccdd"})

Architecture
------------
- **PlaybookAction** — a single step: dispatch, alert, record, classify, monitor,
  predict, geofence, sweep_zone, or check_density.
- **Playbook** — named + ordered list of actions with metadata.
- **PlaybookRunner** — stateful executor that wires actions to the tracking pipeline.
- **PlaybookResult** — outcome of a full playbook execution with per-step results.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("tritium.playbook")


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Types of actions a playbook step can perform."""
    CLASSIFY = "classify"           # Classify a target's threat level
    MONITOR = "monitor"             # Begin monitoring a target or zone
    ALERT = "alert"                 # Fire an alert/notification
    DISPATCH = "dispatch"           # Send an asset to a location
    RECORD = "record"               # Log an observation to history
    PREDICT = "predict"             # Predict target trajectory
    GEOFENCE = "geofence"           # Create or check a geofence zone
    SWEEP_ZONE = "sweep_zone"       # Scan a zone and report findings
    CHECK_DENSITY = "check_density" # Check target density in an area
    WAIT = "wait"                   # Pause for a duration (seconds)


# ---------------------------------------------------------------------------
# PlaybookAction — a single step in a playbook
# ---------------------------------------------------------------------------

@dataclass
class PlaybookAction:
    """A single action within a playbook sequence.

    Attributes
    ----------
    action_type:
        What kind of action to perform.
    name:
        Human-readable label for this step.
    params:
        Action-specific parameters (interpreted by PlaybookRunner).
    condition:
        Optional callable(context) -> bool. If provided and returns False,
        the action is skipped. Allows conditional branching within a playbook.
    on_failure:
        What to do if this action fails: "continue" (default), "abort", or "skip_rest".
    """

    action_type: ActionType
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    condition: Callable[[dict], bool] | None = None
    on_failure: str = "continue"  # "continue", "abort", "skip_rest"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "name": self.name,
            "params": dict(self.params),
            "on_failure": self.on_failure,
            "has_condition": self.condition is not None,
        }


# ---------------------------------------------------------------------------
# StepResult — outcome of executing a single action
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of executing one PlaybookAction."""

    action_name: str
    action_type: str
    success: bool
    skipped: bool = False
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "action_type": self.action_type,
            "success": self.success,
            "skipped": self.skipped,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Playbook — named sequence of actions
# ---------------------------------------------------------------------------

@dataclass
class Playbook:
    """A named, reusable sequence of tactical actions.

    Attributes
    ----------
    playbook_id:
        Unique identifier (e.g., "unknown_entry", "pursuit").
    name:
        Human-readable display name.
    description:
        What this playbook does and when to use it.
    actions:
        Ordered list of PlaybookAction steps.
    tags:
        Categorization tags (e.g., ["threat", "response"]).
    priority:
        Execution priority — higher = more urgent. Default 5.
    """

    playbook_id: str
    name: str
    description: str
    actions: list[PlaybookAction] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    priority: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "playbook_id": self.playbook_id,
            "name": self.name,
            "description": self.description,
            "actions": [a.to_dict() for a in self.actions],
            "tags": list(self.tags),
            "priority": self.priority,
        }


# ---------------------------------------------------------------------------
# PlaybookResult — outcome of a full playbook execution
# ---------------------------------------------------------------------------

@dataclass
class PlaybookResult:
    """Outcome of executing a complete playbook."""

    result_id: str
    playbook_id: str
    playbook_name: str
    success: bool
    steps: list[StepResult] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    completed_at: float = 0.0
    aborted: bool = False
    abort_reason: str = ""

    @property
    def duration(self) -> float:
        if self.completed_at > 0 and self.started_at > 0:
            return self.completed_at - self.started_at
        return 0.0

    @property
    def steps_succeeded(self) -> int:
        return sum(1 for s in self.steps if s.success and not s.skipped)

    @property
    def steps_failed(self) -> int:
        return sum(1 for s in self.steps if not s.success and not s.skipped)

    @property
    def steps_skipped(self) -> int:
        return sum(1 for s in self.steps if s.skipped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "playbook_id": self.playbook_id,
            "playbook_name": self.playbook_name,
            "success": self.success,
            "steps": [s.to_dict() for s in self.steps],
            "context": self.context,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "steps_succeeded": self.steps_succeeded,
            "steps_failed": self.steps_failed,
            "steps_skipped": self.steps_skipped,
        }


# ---------------------------------------------------------------------------
# PlaybookRunner — executes playbooks against the tracking pipeline
# ---------------------------------------------------------------------------

class PlaybookRunner:
    """Executes playbooks against the live tracking pipeline.

    Wires PlaybookAction steps to real TargetTracker, GeofenceEngine,
    AlertEngine, and prediction subsystems.

    Parameters
    ----------
    tracker:
        TargetTracker instance for target queries and updates.
    geofence:
        GeofenceEngine instance for zone operations.
    event_bus:
        EventBus for publishing playbook events.
    alert_engine:
        Optional AlertEngine for firing alerts.
    """

    def __init__(
        self,
        tracker=None,
        geofence=None,
        event_bus=None,
        alert_engine=None,
    ) -> None:
        self._tracker = tracker
        self._geofence = geofence
        self._event_bus = event_bus
        self._alert_engine = alert_engine

        self._playbooks: dict[str, Playbook] = {}
        self._history: list[PlaybookResult] = []
        self._max_history = 1000

        # Action handlers — maps ActionType to executor method
        self._handlers: dict[ActionType, Callable] = {
            ActionType.CLASSIFY: self._execute_classify,
            ActionType.MONITOR: self._execute_monitor,
            ActionType.ALERT: self._execute_alert,
            ActionType.DISPATCH: self._execute_dispatch,
            ActionType.RECORD: self._execute_record,
            ActionType.PREDICT: self._execute_predict,
            ActionType.GEOFENCE: self._execute_geofence,
            ActionType.SWEEP_ZONE: self._execute_sweep_zone,
            ActionType.CHECK_DENSITY: self._execute_check_density,
            ActionType.WAIT: self._execute_wait,
        }

    # ------------------------------------------------------------------
    # Playbook CRUD
    # ------------------------------------------------------------------

    def register_playbook(self, playbook: Playbook) -> None:
        """Register a playbook for later execution."""
        self._playbooks[playbook.playbook_id] = playbook
        logger.info(
            "Playbook registered: %s (%s), %d actions",
            playbook.name,
            playbook.playbook_id,
            len(playbook.actions),
        )

    def unregister_playbook(self, playbook_id: str) -> bool:
        """Remove a playbook. Returns True if found."""
        return self._playbooks.pop(playbook_id, None) is not None

    def get_playbook(self, playbook_id: str) -> Playbook | None:
        """Get a playbook by ID."""
        return self._playbooks.get(playbook_id)

    def list_playbooks(self) -> list[Playbook]:
        """Return all registered playbooks, sorted by priority (highest first)."""
        return sorted(self._playbooks.values(), key=lambda p: p.priority, reverse=True)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        playbook_id: str,
        context: dict[str, Any] | None = None,
    ) -> PlaybookResult:
        """Execute a playbook by ID.

        Parameters
        ----------
        playbook_id:
            ID of the registered playbook to run.
        context:
            Mutable dict of runtime data passed to each action. Actions can
            read from and write to this dict. Common keys: target_id, zone_id,
            position, threshold.

        Returns
        -------
        PlaybookResult with per-step outcomes.
        """
        playbook = self._playbooks.get(playbook_id)
        if playbook is None:
            return PlaybookResult(
                result_id=uuid.uuid4().hex[:12],
                playbook_id=playbook_id,
                playbook_name="<unknown>",
                success=False,
                aborted=True,
                abort_reason=f"Playbook '{playbook_id}' not registered",
                started_at=time.time(),
                completed_at=time.time(),
            )

        return self.execute_playbook(playbook, context)

    def execute_playbook(
        self,
        playbook: Playbook,
        context: dict[str, Any] | None = None,
    ) -> PlaybookResult:
        """Execute a playbook object directly (does not need to be registered).

        Parameters
        ----------
        playbook:
            The Playbook to execute.
        context:
            Mutable runtime context dict.

        Returns
        -------
        PlaybookResult with per-step outcomes.
        """
        ctx = dict(context) if context else {}
        result = PlaybookResult(
            result_id=uuid.uuid4().hex[:12],
            playbook_id=playbook.playbook_id,
            playbook_name=playbook.name,
            success=False,
            context=ctx,
            started_at=time.time(),
        )

        logger.info(
            "Executing playbook: %s (%s), %d actions",
            playbook.name,
            playbook.playbook_id,
            len(playbook.actions),
        )

        self._publish("playbook:started", {
            "playbook_id": playbook.playbook_id,
            "playbook_name": playbook.name,
            "context": ctx,
        })

        for i, action in enumerate(playbook.actions):
            # Check condition
            if action.condition is not None:
                try:
                    if not action.condition(ctx):
                        step = StepResult(
                            action_name=action.name,
                            action_type=action.action_type.value,
                            success=True,
                            skipped=True,
                            data={"reason": "condition_false"},
                        )
                        result.steps.append(step)
                        continue
                except Exception as exc:
                    step = StepResult(
                        action_name=action.name,
                        action_type=action.action_type.value,
                        success=False,
                        skipped=True,
                        error=f"Condition check failed: {exc}",
                    )
                    result.steps.append(step)
                    continue

            # Execute action
            handler = self._handlers.get(action.action_type)
            if handler is None:
                step = StepResult(
                    action_name=action.name,
                    action_type=action.action_type.value,
                    success=False,
                    error=f"No handler for action type: {action.action_type.value}",
                )
                result.steps.append(step)
                if action.on_failure == "abort":
                    result.aborted = True
                    result.abort_reason = f"Step {i} ({action.name}) has no handler"
                    break
                elif action.on_failure == "skip_rest":
                    break
                continue

            try:
                step_data = handler(action, ctx)
                step = StepResult(
                    action_name=action.name,
                    action_type=action.action_type.value,
                    success=True,
                    data=step_data if isinstance(step_data, dict) else {},
                )
            except Exception as exc:
                step = StepResult(
                    action_name=action.name,
                    action_type=action.action_type.value,
                    success=False,
                    error=str(exc),
                )
                if action.on_failure == "abort":
                    result.steps.append(step)
                    result.aborted = True
                    result.abort_reason = f"Step {i} ({action.name}) failed: {exc}"
                    break
                elif action.on_failure == "skip_rest":
                    result.steps.append(step)
                    break

            result.steps.append(step)

        result.completed_at = time.time()
        result.context = ctx
        result.success = not result.aborted and all(
            s.success for s in result.steps if not s.skipped
        )

        self._record_result(result)

        self._publish("playbook:completed", {
            "playbook_id": playbook.playbook_id,
            "result_id": result.result_id,
            "success": result.success,
            "steps_succeeded": result.steps_succeeded,
            "steps_failed": result.steps_failed,
            "duration": result.duration,
        })

        logger.info(
            "Playbook %s completed: success=%s, %d/%d steps ok (%.2fs)",
            playbook.playbook_id,
            result.success,
            result.steps_succeeded,
            len(result.steps),
            result.duration,
        )

        return result

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(self, limit: int = 50) -> list[PlaybookResult]:
        """Return recent playbook execution results (newest first)."""
        return list(reversed(self._history[-limit:]))

    def clear_history(self) -> int:
        """Clear execution history. Returns count removed."""
        count = len(self._history)
        self._history.clear()
        return count

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _execute_classify(self, action: PlaybookAction, ctx: dict) -> dict:
        """Classify a target's threat level using the escalation subsystem."""
        target_id = action.params.get("target_id") or ctx.get("target_id", "")
        if not target_id:
            raise ValueError("classify requires target_id in params or context")

        target = self._get_target(target_id)
        if target is None:
            ctx["classification"] = "unknown"
            ctx["classification_reason"] = "target_not_found"
            return {"target_id": target_id, "classification": "unknown", "reason": "target_not_found"}

        # Use escalation logic if available, enriched with alliance data
        try:
            from tritium_lib.tracking.escalation import (
                ThreatRecord, classify_target, find_zone, EscalationConfig,
            )
            zones = action.params.get("zones", ctx.get("zones", []))
            record = ThreatRecord(target_id=target_id)
            current_zone = find_zone(target.position, zones) if zones else None
            result, _ = classify_target(record, current_zone, time.monotonic())
            classification = result.record.threat_level
            # Escalation only works with zones — if no zone context and the
            # level stayed at "none", fall back to alliance-based classification
            if classification == "none" and target.alliance in ("hostile", "unknown"):
                classification = target.alliance
        except Exception:
            # Fallback: classify based on alliance
            classification = "hostile" if target.alliance == "hostile" else target.alliance

        ctx["classification"] = classification
        ctx["target_alliance"] = target.alliance
        ctx["target_position"] = target.position

        return {
            "target_id": target_id,
            "classification": classification,
            "alliance": target.alliance,
            "position": target.position,
        }

    def _execute_monitor(self, action: PlaybookAction, ctx: dict) -> dict:
        """Begin monitoring a target — record its current state."""
        target_id = action.params.get("target_id") or ctx.get("target_id", "")
        duration = action.params.get("duration", 0)

        target = self._get_target(target_id) if target_id else None

        monitor_data = {
            "target_id": target_id,
            "monitoring": True,
            "duration": duration,
            "timestamp": time.time(),
        }

        if target is not None:
            monitor_data.update({
                "position": target.position,
                "alliance": target.alliance,
                "asset_type": target.asset_type,
                "speed": target.speed,
                "heading": target.heading,
                "confidence": target.effective_confidence,
                "source": target.source,
                "signal_count": target.signal_count,
            })

        ctx["monitor_active"] = True
        ctx["monitor_data"] = monitor_data

        self._publish("playbook:monitor", monitor_data)

        return monitor_data

    def _execute_alert(self, action: PlaybookAction, ctx: dict) -> dict:
        """Fire an alert through the AlertEngine or EventBus."""
        severity = action.params.get("severity", "warning")
        message = action.params.get("message", "Playbook alert triggered")
        target_id = action.params.get("target_id") or ctx.get("target_id", "")

        # Template substitution from context
        try:
            message = message.format(**ctx)
        except (KeyError, IndexError):
            pass

        alert_data = {
            "severity": severity,
            "message": message,
            "target_id": target_id,
            "source": "playbook",
            "timestamp": time.time(),
        }

        if self._alert_engine is not None:
            try:
                self._alert_engine.evaluate_event("target.threat_level.changed", {
                    "target_id": target_id,
                    "severity": severity,
                    "detail": message,
                })
            except Exception:
                logger.debug("AlertEngine evaluation failed", exc_info=True)

        self._publish("playbook:alert", alert_data)
        ctx["alert_fired"] = True
        ctx["last_alert"] = alert_data

        return alert_data

    def _execute_dispatch(self, action: PlaybookAction, ctx: dict) -> dict:
        """Dispatch an asset to a target location."""
        asset_id = action.params.get("asset_id", "")
        target_position = action.params.get("position") or ctx.get("target_position")
        target_id = action.params.get("target_id") or ctx.get("target_id", "")

        if target_position is None and target_id:
            target = self._get_target(target_id)
            if target is not None:
                target_position = target.position

        dispatch_data = {
            "asset_id": asset_id,
            "target_id": target_id,
            "position": target_position,
            "dispatched": True,
            "timestamp": time.time(),
        }

        self._publish("playbook:dispatch", dispatch_data)
        ctx["dispatched"] = True
        ctx["dispatch_data"] = dispatch_data

        return dispatch_data

    def _execute_record(self, action: PlaybookAction, ctx: dict) -> dict:
        """Record an observation to the tracking history."""
        target_id = action.params.get("target_id") or ctx.get("target_id", "")
        note = action.params.get("note", "")
        data = dict(action.params.get("data", {}))

        record_entry = {
            "target_id": target_id,
            "note": note,
            "data": data,
            "context_snapshot": {
                k: v for k, v in ctx.items()
                if isinstance(v, (str, int, float, bool, list, tuple))
            },
            "timestamp": time.time(),
        }

        # Record to target history if tracker available
        if self._tracker is not None and target_id:
            target = self._get_target(target_id)
            if target is not None:
                self._tracker.history.record(target_id, target.position)

        self._publish("playbook:record", record_entry)
        ctx.setdefault("records", []).append(record_entry)

        return record_entry

    def _execute_predict(self, action: PlaybookAction, ctx: dict) -> dict:
        """Predict target trajectory using the prediction subsystem."""
        target_id = action.params.get("target_id") or ctx.get("target_id", "")
        horizons = action.params.get("horizons", [1, 5, 15])

        if not target_id:
            raise ValueError("predict requires target_id in params or context")

        predictions = []
        if self._tracker is not None:
            try:
                from tritium_lib.tracking.target_prediction import predict_target
                preds = predict_target(
                    target_id,
                    self._tracker.history,
                    horizons=horizons,
                )
                predictions = [p.to_dict() for p in preds]
            except Exception:
                logger.debug("Prediction failed for %s", target_id, exc_info=True)

        predict_data = {
            "target_id": target_id,
            "horizons": horizons,
            "predictions": predictions,
            "has_predictions": len(predictions) > 0,
        }

        ctx["predictions"] = predictions
        ctx["predicted"] = len(predictions) > 0

        return predict_data

    def _execute_geofence(self, action: PlaybookAction, ctx: dict) -> dict:
        """Create a geofence zone or check a target against existing zones."""
        operation = action.params.get("operation", "check")  # "create" or "check"

        if operation == "create":
            return self._geofence_create(action, ctx)
        else:
            return self._geofence_check(action, ctx)

    def _geofence_create(self, action: PlaybookAction, ctx: dict) -> dict:
        """Create a new geofence zone."""
        zone_id = action.params.get("zone_id", uuid.uuid4().hex[:12])
        zone_name = action.params.get("zone_name", "Playbook Zone")
        polygon = action.params.get("polygon", [])
        zone_type = action.params.get("zone_type", "monitored")
        center = action.params.get("center") or ctx.get("target_position")
        radius = action.params.get("radius", 20.0)

        # If no polygon but center+radius given, create a rough octagon
        if not polygon and center is not None:
            cx, cy = center
            polygon = _circle_polygon(cx, cy, radius, sides=8)

        result_data = {
            "operation": "create",
            "zone_id": zone_id,
            "zone_name": zone_name,
            "zone_type": zone_type,
            "polygon": polygon,
            "created": False,
        }

        if self._geofence is not None and polygon:
            try:
                from tritium_lib.tracking.geofence import GeoZone
                zone = GeoZone(
                    zone_id=zone_id,
                    name=zone_name,
                    polygon=[tuple(p) for p in polygon],
                    zone_type=zone_type,
                )
                self._geofence.add_zone(zone)
                result_data["created"] = True
                ctx["zone_id"] = zone_id
                ctx["zone_polygon"] = polygon
            except Exception as exc:
                result_data["error"] = str(exc)

        return result_data

    def _geofence_check(self, action: PlaybookAction, ctx: dict) -> dict:
        """Check a target against geofence zones."""
        target_id = action.params.get("target_id") or ctx.get("target_id", "")
        position = action.params.get("position") or ctx.get("target_position")

        if position is None and target_id:
            target = self._get_target(target_id)
            if target is not None:
                position = target.position

        result_data = {
            "operation": "check",
            "target_id": target_id,
            "position": position,
            "events": [],
            "in_zones": [],
        }

        if self._geofence is not None and position is not None:
            events = self._geofence.check(target_id, tuple(position))
            result_data["events"] = [e.to_dict() for e in events]
            zones = self._geofence.get_target_zones(target_id)
            result_data["in_zones"] = list(zones)
            ctx["in_zones"] = list(zones)
            ctx["geofence_events"] = [e.to_dict() for e in events]

        return result_data

    def _execute_sweep_zone(self, action: PlaybookAction, ctx: dict) -> dict:
        """Sweep a zone — enumerate all targets within its boundaries."""
        zone_id = action.params.get("zone_id") or ctx.get("zone_id", "")
        center = action.params.get("center") or ctx.get("sweep_center")
        radius = action.params.get("radius", 50.0)

        found_targets: list[dict] = []

        if self._tracker is not None:
            all_targets = self._tracker.get_all()

            if zone_id and self._geofence is not None:
                # Use geofence zone membership
                occupants = self._geofence.get_zone_occupants(zone_id)
                for t in all_targets:
                    if t.target_id in occupants:
                        found_targets.append({
                            "target_id": t.target_id,
                            "name": t.name,
                            "alliance": t.alliance,
                            "asset_type": t.asset_type,
                            "position": t.position,
                            "source": t.source,
                        })
            elif center is not None:
                # Radius-based sweep
                cx, cy = center
                for t in all_targets:
                    dx = t.position[0] - cx
                    dy = t.position[1] - cy
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist <= radius:
                        found_targets.append({
                            "target_id": t.target_id,
                            "name": t.name,
                            "alliance": t.alliance,
                            "asset_type": t.asset_type,
                            "position": t.position,
                            "distance": round(dist, 2),
                            "source": t.source,
                        })

        sweep_data = {
            "zone_id": zone_id,
            "center": center,
            "radius": radius,
            "targets_found": len(found_targets),
            "targets": found_targets,
            "hostiles": sum(1 for t in found_targets if t["alliance"] == "hostile"),
            "unknowns": sum(1 for t in found_targets if t["alliance"] == "unknown"),
            "friendlies": sum(1 for t in found_targets if t["alliance"] == "friendly"),
        }

        ctx["sweep_results"] = sweep_data
        ctx["sweep_target_count"] = len(found_targets)
        self._publish("playbook:sweep", sweep_data)

        return sweep_data

    def _execute_check_density(self, action: PlaybookAction, ctx: dict) -> dict:
        """Check target density in an area and compare against threshold."""
        center = action.params.get("center") or ctx.get("target_position")
        radius = action.params.get("radius", 30.0)
        threshold = action.params.get("threshold", 5)

        count = 0
        target_ids: list[str] = []

        if self._tracker is not None and center is not None:
            cx, cy = center
            for t in self._tracker.get_all():
                dx = t.position[0] - cx
                dy = t.position[1] - cy
                if math.sqrt(dx * dx + dy * dy) <= radius:
                    count += 1
                    target_ids.append(t.target_id)

        exceeds = count >= threshold
        density_data = {
            "center": center,
            "radius": radius,
            "count": count,
            "threshold": threshold,
            "exceeds_threshold": exceeds,
            "target_ids": target_ids,
        }

        ctx["density_count"] = count
        ctx["density_exceeds"] = exceeds
        ctx["density_target_ids"] = target_ids

        return density_data

    def _execute_wait(self, action: PlaybookAction, ctx: dict) -> dict:
        """Pause execution (records the wait without actually sleeping)."""
        duration = action.params.get("duration", 0)
        ctx["wait_duration"] = duration
        return {"duration": duration, "waited": True}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_target(self, target_id: str):
        """Get a target from the tracker, returns None if unavailable."""
        if self._tracker is None:
            return None
        return self._tracker.get_target(target_id)

    def _publish(self, topic: str, data: dict) -> None:
        """Publish event if bus is available."""
        if self._event_bus is not None:
            try:
                self._event_bus.publish(topic, data)
            except Exception:
                logger.debug("Event publish failed: %s", topic, exc_info=True)

    def _record_result(self, result: PlaybookResult) -> None:
        """Append to history, trimming if necessary."""
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def _circle_polygon(
    cx: float, cy: float, radius: float, sides: int = 8
) -> list[tuple[float, float]]:
    """Generate a regular polygon approximating a circle."""
    points = []
    for i in range(sides):
        angle = 2.0 * math.pi * i / sides
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((round(x, 4), round(y, 4)))
    return points


# ---------------------------------------------------------------------------
# Built-in playbooks
# ---------------------------------------------------------------------------

def _pb_unknown_entry() -> Playbook:
    """unknown_entry — new target enters area -> classify -> monitor -> alert if threat."""
    return Playbook(
        playbook_id="unknown_entry",
        name="Unknown Entry Response",
        description=(
            "Triggered when a new unknown target enters the area of operations. "
            "Classifies the target, begins monitoring, and fires an alert if "
            "the classification indicates a threat."
        ),
        actions=[
            PlaybookAction(
                action_type=ActionType.CLASSIFY,
                name="Classify incoming target",
                params={},
            ),
            PlaybookAction(
                action_type=ActionType.RECORD,
                name="Record initial observation",
                params={"note": "New target detected, classification initiated"},
            ),
            PlaybookAction(
                action_type=ActionType.MONITOR,
                name="Begin target monitoring",
                params={"duration": 60},
            ),
            PlaybookAction(
                action_type=ActionType.ALERT,
                name="Alert if threat detected",
                params={
                    "severity": "warning",
                    "message": "Unknown entry: target {target_id} classified as {classification}",
                },
                condition=lambda ctx: ctx.get("classification") in ("hostile", "suspicious"),
            ),
        ],
        tags=["threat", "entry", "response"],
        priority=8,
    )


def _pb_pursuit() -> Playbook:
    """pursuit — target leaves zone -> track trajectory -> predict destination."""
    return Playbook(
        playbook_id="pursuit",
        name="Pursuit Protocol",
        description=(
            "Activated when a tracked target exits a monitored zone. "
            "Records the departure, predicts the target's trajectory, "
            "and optionally dispatches an asset to the predicted destination."
        ),
        actions=[
            PlaybookAction(
                action_type=ActionType.RECORD,
                name="Record zone departure",
                params={"note": "Target departed monitored zone, initiating pursuit"},
            ),
            PlaybookAction(
                action_type=ActionType.CLASSIFY,
                name="Re-classify departing target",
                params={},
            ),
            PlaybookAction(
                action_type=ActionType.PREDICT,
                name="Predict target trajectory",
                params={"horizons": [1, 5, 15]},
            ),
            PlaybookAction(
                action_type=ActionType.ALERT,
                name="Alert on hostile departure",
                params={
                    "severity": "critical",
                    "message": "Hostile target {target_id} departed zone — pursuit active",
                },
                condition=lambda ctx: ctx.get("classification") == "hostile",
            ),
            PlaybookAction(
                action_type=ActionType.DISPATCH,
                name="Dispatch asset to predicted position",
                params={},
                condition=lambda ctx: ctx.get("predicted", False),
            ),
        ],
        tags=["pursuit", "tracking", "prediction"],
        priority=9,
    )


def _pb_gathering() -> Playbook:
    """gathering — multiple targets converge -> monitor density -> alert if threshold."""
    return Playbook(
        playbook_id="gathering",
        name="Gathering Detection",
        description=(
            "Monitors for convergence of multiple targets in one area. "
            "Checks target density against a threshold and escalates with "
            "an alert and sweep if the threshold is exceeded."
        ),
        actions=[
            PlaybookAction(
                action_type=ActionType.CHECK_DENSITY,
                name="Check target density",
                params={"radius": 30.0, "threshold": 5},
            ),
            PlaybookAction(
                action_type=ActionType.ALERT,
                name="Alert on gathering detected",
                params={
                    "severity": "warning",
                    "message": "Gathering detected: {density_count} targets within radius",
                },
                condition=lambda ctx: ctx.get("density_exceeds", False),
            ),
            PlaybookAction(
                action_type=ActionType.SWEEP_ZONE,
                name="Sweep gathering area",
                params={"radius": 50.0},
                condition=lambda ctx: ctx.get("density_exceeds", False),
            ),
            PlaybookAction(
                action_type=ActionType.RECORD,
                name="Record gathering event",
                params={"note": "Gathering analysis completed"},
            ),
        ],
        tags=["density", "gathering", "crowd"],
        priority=6,
    )


def _pb_sweep() -> Playbook:
    """sweep — systematic search of an area -> zone-by-zone -> report findings."""
    return Playbook(
        playbook_id="sweep",
        name="Area Sweep",
        description=(
            "Performs a systematic search of an area. Sweeps the zone, "
            "classifies any found targets, records findings, and fires "
            "an alert if hostiles are found."
        ),
        actions=[
            PlaybookAction(
                action_type=ActionType.RECORD,
                name="Begin sweep operation",
                params={"note": "Area sweep initiated"},
            ),
            PlaybookAction(
                action_type=ActionType.SWEEP_ZONE,
                name="Sweep designated area",
                params={"radius": 100.0},
            ),
            PlaybookAction(
                action_type=ActionType.ALERT,
                name="Alert on hostiles found",
                params={
                    "severity": "critical",
                    "message": "Sweep found {sweep_target_count} targets in area",
                },
                condition=lambda ctx: ctx.get("sweep_results", {}).get("hostiles", 0) > 0,
            ),
            PlaybookAction(
                action_type=ActionType.RECORD,
                name="Record sweep findings",
                params={"note": "Sweep completed"},
            ),
        ],
        tags=["sweep", "search", "recon"],
        priority=5,
    )


def _pb_perimeter() -> Playbook:
    """perimeter — establish geofence -> monitor all crossings -> log everything."""
    return Playbook(
        playbook_id="perimeter",
        name="Perimeter Watch",
        description=(
            "Establishes a geofence perimeter around a position, then "
            "checks all known targets against it and records the results. "
            "Alerts on any targets found inside the perimeter."
        ),
        actions=[
            PlaybookAction(
                action_type=ActionType.GEOFENCE,
                name="Establish perimeter zone",
                params={
                    "operation": "create",
                    "zone_name": "Perimeter",
                    "zone_type": "restricted",
                    "radius": 25.0,
                },
            ),
            PlaybookAction(
                action_type=ActionType.GEOFENCE,
                name="Check targets against perimeter",
                params={"operation": "check"},
            ),
            PlaybookAction(
                action_type=ActionType.SWEEP_ZONE,
                name="Sweep perimeter interior",
                params={},
            ),
            PlaybookAction(
                action_type=ActionType.RECORD,
                name="Log perimeter status",
                params={"note": "Perimeter established and initial sweep complete"},
            ),
            PlaybookAction(
                action_type=ActionType.ALERT,
                name="Alert on perimeter breach",
                params={
                    "severity": "critical",
                    "message": "Perimeter active: {sweep_target_count} targets inside zone",
                },
                condition=lambda ctx: ctx.get("sweep_target_count", 0) > 0,
            ),
        ],
        tags=["perimeter", "geofence", "security"],
        priority=7,
    )


# Registry of all built-in playbooks
BUILTIN_PLAYBOOKS: dict[str, Callable[[], Playbook]] = {
    "unknown_entry": _pb_unknown_entry,
    "pursuit": _pb_pursuit,
    "gathering": _pb_gathering,
    "sweep": _pb_sweep,
    "perimeter": _pb_perimeter,
}


def load_builtin_playbooks() -> list[Playbook]:
    """Instantiate and return all built-in playbooks."""
    return [factory() for factory in BUILTIN_PLAYBOOKS.values()]


__all__ = [
    "ActionType",
    "PlaybookAction",
    "StepResult",
    "Playbook",
    "PlaybookResult",
    "PlaybookRunner",
    "BUILTIN_PLAYBOOKS",
    "load_builtin_playbooks",
]
