# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Threat escalation domain logic — threat level classification and zone matching.

Provides the core data model and pure-logic functions for threat classification:
  - ThreatRecord: per-target threat state
  - THREAT_LEVELS: the escalation ladder (none -> unknown -> suspicious -> hostile)
  - Zone containment checks (radius-based)
  - Escalation / de-escalation tick logic

This module is framework-agnostic. It does NOT depend on EventBus, MQTT,
SimulationEngine, or any other SC infrastructure. The SC module
``engine.tactical.escalation`` wraps this with EventBus subscriptions,
threading, and auto-dispatch.

Usage
-----
    from tritium_lib.tracking.escalation import (
        ThreatRecord, THREAT_LEVELS, EscalationConfig,
        classify_target, find_zone, escalation_index,
    )

    record = ThreatRecord(target_id="ble_AA:BB:CC")
    zone = find_zone((10.0, 20.0), zones)
    record = classify_target(record, zone, now, config=EscalationConfig())
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

# Threat ladder levels (ordered lowest to highest)
THREAT_LEVELS = ["none", "unknown", "suspicious", "hostile"]


def escalation_index(level: str) -> int:
    """Return the numeric index of a threat level.

    Returns 0 for unknown/invalid levels.
    """
    try:
        return THREAT_LEVELS.index(level)
    except ValueError:
        return 0


def is_escalation(old_level: str, new_level: str) -> bool:
    """Return True if new_level is higher than old_level on the threat ladder."""
    return escalation_index(new_level) > escalation_index(old_level)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EscalationConfig:
    """Tunable parameters for escalation / de-escalation logic.

    Defaults match the original SC ThreatClassifier constants.
    """
    linger_threshold: float = 30.0    # seconds in zone before -> hostile
    deescalation_time: float = 30.0   # seconds outside zones before step-down
    tick_interval: float = 0.5        # recommended evaluation frequency (2 Hz)
    # Passive decay — drops threat level by one band every N seconds when
    # the level has been stable (no new escalation) regardless of whether
    # the target is still inside a zone.  Closes Gap-fix C M-5 where
    # Behavioral SAT measured a target pinned at "hostile" for 9626s.
    # Set to 0.0 to disable passive decay (zone-exit decay still applies).
    passive_decay_interval: float = 60.0


# ---------------------------------------------------------------------------
# ThreatRecord — per-target threat state
# ---------------------------------------------------------------------------

@dataclass
class ThreatRecord:
    """Tracks threat state for a single target.

    One record exists per non-friendly, non-neutral target that the
    classifier has seen.  Records persist even when threat_level returns
    to ``none`` (they are only pruned when the target disappears from
    the TargetTracker).
    """

    target_id: str
    threat_level: str = "none"
    level_since: float = field(default_factory=time.monotonic)
    in_zone: str = ""         # name of the zone the target is currently in
    zone_enter_time: float = 0.0
    last_update: float = field(default_factory=time.monotonic)
    prior_hostile: bool = False   # was this target ever classified as hostile?

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary."""
        return {
            "target_id": self.target_id,
            "threat_level": self.threat_level,
            "level_since": self.level_since,
            "in_zone": self.in_zone,
            "zone_enter_time": self.zone_enter_time,
            "last_update": self.last_update,
            "prior_hostile": self.prior_hostile,
        }


# ---------------------------------------------------------------------------
# Zone matching
# ---------------------------------------------------------------------------

def find_zone(
    position: tuple[float, float],
    zones: list[dict],
) -> Optional[dict]:
    """Find the most restrictive zone containing the given position.

    Uses simple radius-based containment: a zone contains a point if the
    point is within the zone's radius (default 10 units) of the zone center.
    When a position is inside multiple zones, restricted zones take priority.

    Parameters
    ----------
    position:
        ``(x, y)`` coordinates to test.
    zones:
        List of zone dicts, each with ``position`` (``{x, z}`` or ``{x, y}``),
        ``type`` (e.g. ``"restricted_area"``), ``name``, and optionally
        ``properties.radius``.

    Returns
    -------
    The matching zone dict, or *None* if no zone contains the position.
    """
    px, py = position
    best: Optional[dict] = None
    for zone in zones:
        zpos = zone.get("position", {})
        zx = zpos.get("x", 0.0)
        zy = zpos.get("z", zpos.get("y", 0.0))
        radius = zone.get("properties", {}).get("radius", 10.0)
        dist = math.hypot(px - zx, py - zy)
        if dist <= radius:
            if "restricted" in zone.get("type", ""):
                return zone  # restricted always wins
            if best is None:
                best = zone
    return best


# ---------------------------------------------------------------------------
# Escalation tick — pure function (no side effects)
# ---------------------------------------------------------------------------

@dataclass
class ClassifyResult:
    """Result of a single classify_target call.

    Attributes
    ----------
    record:
        The updated ThreatRecord.
    level_changed:
        True if the threat level changed during this tick.
    old_level:
        Previous threat level (only meaningful when *level_changed* is True).
    new_level:
        Current threat level after classification.
    reason:
        Human-readable reason string for the level change.
    zone_entered:
        Non-empty if the target entered a new zone this tick.
    """
    record: ThreatRecord
    level_changed: bool = False
    old_level: str = "none"
    new_level: str = "none"
    reason: str = ""
    zone_entered: str = ""


def classify_target(
    record: ThreatRecord,
    current_zone: Optional[dict],
    now: float,
    config: Optional[EscalationConfig] = None,
    zone_exit_time: float = 0.0,
) -> tuple[ClassifyResult, float]:
    """Apply one tick of escalation / de-escalation logic to a target.

    This is the pure-function equivalent of
    ``ThreatClassifier._classify_tick()`` in SC.  It updates the
    *record* in-place and returns a ``ClassifyResult`` describing what
    changed plus the updated zone-exit timestamp.

    Parameters
    ----------
    record:
        The target's current ThreatRecord (modified in-place).
    current_zone:
        The zone dict the target is currently inside, or None.
    now:
        Current monotonic timestamp.
    config:
        Escalation timing parameters. Defaults to standard config.
    zone_exit_time:
        Monotonic time when the target last left all zones (0 if still in a zone).

    Returns
    -------
    A 2-tuple of (ClassifyResult, updated_zone_exit_time).
    """
    cfg = config or EscalationConfig()
    old_level = record.threat_level
    zone_entered = ""

    record.last_update = now

    if current_zone is not None:
        zone_type = current_zone.get("type", "")
        zone_name = current_zone.get("name", zone_type) or "<unnamed>"

        # Track zone entry
        if record.in_zone != zone_name:
            record.in_zone = zone_name
            record.zone_enter_time = now
            zone_exit_time = 0.0
            zone_entered = zone_name

        # Escalation based on zone type
        if "restricted" in zone_type:
            if escalation_index(record.threat_level) < escalation_index("suspicious"):
                record.threat_level = "suspicious"
                record.level_since = now
        elif record.threat_level == "none":
            # Prior hostiles skip unknown — they've earned suspicion
            if record.prior_hostile:
                record.threat_level = "suspicious"
            else:
                record.threat_level = "unknown"
            record.level_since = now

        # Linger escalation
        time_in_zone = now - record.zone_enter_time
        if time_in_zone > cfg.linger_threshold:
            if escalation_index(record.threat_level) < escalation_index("hostile"):
                record.threat_level = "hostile"
                record.level_since = now
                record.prior_hostile = True

    else:
        # Target outside all zones — track for de-escalation
        if record.in_zone:
            record.in_zone = ""
            zone_exit_time = now

        # De-escalation after time outside zones
        if zone_exit_time > 0 and (now - zone_exit_time) > cfg.deescalation_time:
            level_idx = escalation_index(record.threat_level)
            if level_idx > 0:
                record.threat_level = THREAT_LEVELS[level_idx - 1]
                record.level_since = now
                zone_exit_time = now  # reset timer for next step

    # Passive decay — fires regardless of zone presence so a target pinned
    # at "hostile" inside a restricted zone can still drop after the
    # configured interval if no fresh escalation event has fired.  Only
    # applies when the level has not just been raised this tick.
    # (Closes Gap-fix C M-5.)
    decay_eligible = (
        cfg.passive_decay_interval > 0.0
        and record.threat_level == old_level   # not changed by zone logic this tick
        and escalation_index(record.threat_level) > 0
        and (now - record.level_since) >= cfg.passive_decay_interval
    )
    if decay_eligible:
        level_idx = escalation_index(record.threat_level)
        record.threat_level = THREAT_LEVELS[level_idx - 1]
        record.level_since = now

    # Build result
    level_changed = record.threat_level != old_level
    reason = ""
    if level_changed:
        if decay_eligible and not record.in_zone:
            reason = "passive-decay"
        elif decay_eligible and record.in_zone:
            reason = f"passive-decay-in-zone:{record.in_zone}"
        else:
            reason = f"zone:{record.in_zone}" if record.in_zone else "de-escalation"

    result = ClassifyResult(
        record=record,
        level_changed=level_changed,
        old_level=old_level,
        new_level=record.threat_level,
        reason=reason,
        zone_entered=zone_entered,
    )
    return result, zone_exit_time


# ---------------------------------------------------------------------------
# Batch classification helper
# ---------------------------------------------------------------------------

def classify_all_targets(
    records: dict[str, ThreatRecord],
    zone_exit_times: dict[str, float],
    targets: list,
    zones: list[dict],
    now: float,
    config: Optional[EscalationConfig] = None,
) -> list[ClassifyResult]:
    """Classify a batch of targets and return all results.

    Prunes records for targets no longer in the ``targets`` list.
    Creates new ThreatRecord entries for non-friendly, non-neutral
    targets that appear for the first time.

    Parameters
    ----------
    records:
        Mutable dict of target_id -> ThreatRecord (updated in-place).
    zone_exit_times:
        Mutable dict of target_id -> float (updated in-place).
    targets:
        All tracked targets.  Each must have ``target_id``, ``alliance``,
        and ``position`` attributes (duck-typed).
    zones:
        Zone definitions for find_zone().
    now:
        Current monotonic timestamp.
    config:
        Escalation timing parameters.

    Returns
    -------
    List of ClassifyResult for targets whose level changed or who entered
    a new zone (zone_entered is non-empty).
    """
    cfg = config or EscalationConfig()
    results: list[ClassifyResult] = []

    # Prune records for targets no longer tracked
    tracked_ids = set()
    for t in targets:
        tid = getattr(t, "target_id", None) or (
            t.get("target_id", "") if isinstance(t, dict) else ""
        )
        if tid:
            tracked_ids.add(tid)

    stale = [tid for tid in records if tid not in tracked_ids]
    for tid in stale:
        del records[tid]
        zone_exit_times.pop(tid, None)

    for target in targets:
        # Duck-type target access
        if hasattr(target, "target_id"):
            tid = target.target_id
            alliance = target.alliance
            position = target.position
        elif isinstance(target, dict):
            tid = target.get("target_id", "")
            alliance = target.get("alliance", "")
            pos_data = target.get("position", (0.0, 0.0))
            if isinstance(pos_data, dict):
                position = (pos_data.get("x", 0.0), pos_data.get("y", 0.0))
            else:
                position = pos_data
        else:
            continue

        if not tid:
            continue

        # Skip friendly and neutral targets
        if alliance in ("friendly", "neutral"):
            continue

        # Get or create record
        if tid not in records:
            records[tid] = ThreatRecord(target_id=tid)
        record = records[tid]

        # Find zone and classify
        current_zone = find_zone(position, zones)
        exit_time = zone_exit_times.get(tid, 0.0)
        result, new_exit = classify_target(record, current_zone, now, cfg, exit_time)

        if new_exit != exit_time:
            if new_exit > 0:
                zone_exit_times[tid] = new_exit
            else:
                zone_exit_times.pop(tid, None)

        if result.level_changed or result.zone_entered:
            results.append(result)

    return results
