# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Canonical target status enumeration and lifecycle groupings.

This module is the single source of truth for the string values that
appear in :attr:`tritium_lib.sim_engine.core.entity.SimulationTarget.status`
and in every consumer that filters or groups targets by lifecycle phase.

Lifecycle (W201-sanctioned)
---------------------------

Every target moves through a sequence of statuses:

    active -> {idle, stationary, arrived, low_battery, ...} -> <terminal>

The five **terminal** statuses below mark a target as no-longer-active.
The frontend header counter, the REST ``/api/targets`` endpoint, and the
WebSocket broadcast bridge all skip targets in these states so the
header/API counts stay aligned (Wave 198 mismatch fix, hardened by the
W201 single-source-of-truth refactor).

Subsystem ownership
-------------------

* ``active`` — set by :class:`SimulationTarget` at construction time and
  by friendly dispatch logic when a unit returns to action.
* ``idle`` — set by friendly behaviour FSMs when a unit is between tasks.
* ``stationary`` — set by turret behaviour and by parked vehicle logic.
* ``arrived`` — set by friendly dispatcher when a one-shot waypoint route
  completes (``loop_waypoints=False``).
* ``low_battery`` — set by the battery drain pass for persons/animals.
* ``escaped`` (terminal) — set by hostile pathing when a hostile reaches
  an exit edge of the simulation map.
* ``neutralized`` (terminal) — set by combat resolution when a hostile is
  intercepted by a friendly unit.
* ``eliminated`` (terminal) — set by combat resolution when any
  combatant takes damage past zero health.
* ``destroyed`` (terminal) — set by the engine for inanimate assets
  (vehicles, structures) that have been wrecked.
* ``despawned`` (terminal) — set by neutral routing logic when a neutral
  reaches its destination and is removed from the active roster.

Cross-language contract
-----------------------

The set of terminal statuses is also mirrored verbatim in
``tritium-lib/web/utils.js`` (``TERMINAL_STATUSES``) so the JS frontend
agrees with the Python backend.  Adding or removing a terminal status
requires updating both files plus the regression test in
``tests/models/test_target_status.py``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical status values
# ---------------------------------------------------------------------------

#: Every status string the simulation engine may assign to a target.
#: Kept as a frozenset so it can be used as a default argument and
#: iterated cheaply in hot paths without risk of mutation.
ALL_STATUSES: frozenset[str] = frozenset({
    "active",
    "idle",
    "stationary",
    "arrived",
    "low_battery",
    "escaped",
    "neutralized",
    "eliminated",
    "destroyed",
    "despawned",
})

#: Statuses that mean the target is no longer active.  Targets in these
#: states are excluded from ``/api/targets`` (unless ``include_terminal``
#: is true) and from the frontend header counter so the two counts match.
#:
#: This set is a strict subset of :data:`ALL_STATUSES` and must remain
#: exactly five elements: see Wave 198/201 history.  Tests in
#: ``tests/models/test_target_status.py`` enforce this invariant.
TERMINAL_STATUSES: frozenset[str] = frozenset({
    "eliminated",
    "destroyed",
    "despawned",
    "neutralized",
    "escaped",
})

# ---------------------------------------------------------------------------
# Lifecycle groupings
# ---------------------------------------------------------------------------

#: Statuses where the target is alive and present but not actively moving
#: along a waypoint route.  Useful for "reachable but quiet" filters.
RESTING_STATUSES: frozenset[str] = frozenset({
    "idle",
    "stationary",
    "arrived",
})

#: Statuses where the target is actively progressing through behaviour
#: (moving, fighting, scanning, draining battery).
ACTIVE_STATUSES: frozenset[str] = frozenset({
    "active",
    "low_battery",
})


def is_terminal(status: object) -> bool:
    """Return ``True`` if *status* is one of the terminal lifecycle states.

    Defensive against non-string inputs so callers pulling status off
    arbitrary dict-shaped batches do not need to pre-validate.
    """
    if not isinstance(status, str):
        return False
    return status in TERMINAL_STATUSES


__all__ = [
    "ALL_STATUSES",
    "TERMINAL_STATUSES",
    "RESTING_STATUSES",
    "ACTIVE_STATUSES",
    "is_terminal",
]
