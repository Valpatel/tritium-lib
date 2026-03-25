# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Data retention policies and enforcement.

Defines how long different categories of data are kept, and provides
a ``RetentionManager`` that enforces those policies by identifying and
purging expired records.

Default retention windows:
    - realtime_sightings : 7 days
    - target_history     : 30 days
    - dossiers           : 90 days
    - incidents          : 365 days
    - audit_trail        : 2555 days  (7 years)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger("privacy.retention")

# ---------------------------------------------------------------------------
# Data categories
# ---------------------------------------------------------------------------

class DataCategory(str, Enum):
    """Categories of data subject to retention policies."""
    REALTIME_SIGHTINGS = "realtime_sightings"
    TARGET_HISTORY = "target_history"
    DOSSIERS = "dossiers"
    INCIDENTS = "incidents"
    AUDIT_TRAIL = "audit_trail"
    CAMERA_FRAMES = "camera_frames"
    SENSOR_TELEMETRY = "sensor_telemetry"
    CHAT_MESSAGES = "chat_messages"
    LOCATION_TRACKS = "location_tracks"


# ---------------------------------------------------------------------------
# Default retention periods (seconds)
# ---------------------------------------------------------------------------

_SECONDS_PER_DAY = 86400

DEFAULT_RETENTION: dict[str, int] = {
    DataCategory.REALTIME_SIGHTINGS: 7 * _SECONDS_PER_DAY,
    DataCategory.TARGET_HISTORY: 30 * _SECONDS_PER_DAY,
    DataCategory.DOSSIERS: 90 * _SECONDS_PER_DAY,
    DataCategory.INCIDENTS: 365 * _SECONDS_PER_DAY,
    DataCategory.AUDIT_TRAIL: 7 * 365 * _SECONDS_PER_DAY,
    DataCategory.CAMERA_FRAMES: 3 * _SECONDS_PER_DAY,
    DataCategory.SENSOR_TELEMETRY: 14 * _SECONDS_PER_DAY,
    DataCategory.CHAT_MESSAGES: 30 * _SECONDS_PER_DAY,
    DataCategory.LOCATION_TRACKS: 30 * _SECONDS_PER_DAY,
}


# ---------------------------------------------------------------------------
# RetentionPolicy — single-category configuration
# ---------------------------------------------------------------------------

@dataclass
class RetentionPolicy:
    """Retention rules for one data category.

    Parameters
    ----------
    category :
        Which data category this policy governs.
    retention_seconds :
        How long records are kept.  Records older than ``now - retention_seconds``
        are eligible for purge.
    enabled :
        If False, this policy is not enforced (data is kept indefinitely).
    description :
        Human-readable explanation of the policy.
    legal_basis :
        GDPR legal basis for processing (e.g. "legitimate_interest", "consent").
    """

    category: str
    retention_seconds: int
    enabled: bool = True
    description: str = ""
    legal_basis: str = "legitimate_interest"
    created_at: float = field(default_factory=time.time)

    @property
    def retention_days(self) -> float:
        """Retention period expressed in days."""
        return self.retention_seconds / _SECONDS_PER_DAY

    def is_expired(self, record_timestamp: float, now: Optional[float] = None) -> bool:
        """Return True if a record with the given timestamp has expired."""
        if not self.enabled:
            return False
        now = now if now is not None else time.time()
        return (now - record_timestamp) > self.retention_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict."""
        return {
            "category": self.category,
            "retention_seconds": self.retention_seconds,
            "retention_days": self.retention_days,
            "enabled": self.enabled,
            "description": self.description,
            "legal_basis": self.legal_basis,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# PurgeResult — what got deleted
# ---------------------------------------------------------------------------

@dataclass
class PurgeResult:
    """Result of a retention enforcement run."""

    category: str
    purged_count: int
    cutoff_timestamp: float
    execution_time: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "purged_count": self.purged_count,
            "cutoff_timestamp": self.cutoff_timestamp,
            "execution_time": self.execution_time,
            "errors": list(self.errors),
            "success": self.success,
        }


# ---------------------------------------------------------------------------
# PurgeHandler type
# ---------------------------------------------------------------------------

PurgeHandler = Callable[[str, float], int]
"""Callable(category, cutoff_timestamp) -> count of records purged."""


# ---------------------------------------------------------------------------
# RetentionManager — enforce policies
# ---------------------------------------------------------------------------

class RetentionManager:
    """Enforce data retention policies by purging expired records.

    Register a ``PurgeHandler`` for each category.  When ``enforce()`` is
    called, the manager computes the cutoff timestamp for each active
    policy and invokes the handler to delete records older than that cutoff.

    Usage
    -----
    ::

        mgr = RetentionManager()
        mgr.register_handler("realtime_sightings", my_purge_fn)
        results = mgr.enforce()

    Parameters
    ----------
    policies :
        Optional dict of ``{category: RetentionPolicy}``.  If not supplied,
        defaults are loaded from :data:`DEFAULT_RETENTION`.
    """

    def __init__(
        self,
        policies: Optional[dict[str, RetentionPolicy]] = None,
    ) -> None:
        if policies is not None:
            self._policies: dict[str, RetentionPolicy] = dict(policies)
        else:
            self._policies = {
                cat: RetentionPolicy(
                    category=cat,
                    retention_seconds=secs,
                    description=f"Default {cat} retention",
                )
                for cat, secs in DEFAULT_RETENTION.items()
            }
        self._handlers: dict[str, PurgeHandler] = {}
        self._history: list[PurgeResult] = []

    # -- policy management --------------------------------------------------

    def get_policy(self, category: str) -> Optional[RetentionPolicy]:
        """Return the policy for *category*, or None."""
        return self._policies.get(category)

    def set_policy(self, policy: RetentionPolicy) -> None:
        """Add or replace a retention policy."""
        self._policies[policy.category] = policy
        logger.info("Retention policy set: %s = %d days",
                     policy.category, policy.retention_days)

    def remove_policy(self, category: str) -> bool:
        """Remove a policy.  Returns True if it existed."""
        return self._policies.pop(category, None) is not None

    def list_policies(self) -> list[RetentionPolicy]:
        """Return all current policies sorted by category name."""
        return sorted(self._policies.values(), key=lambda p: p.category)

    # -- handler management -------------------------------------------------

    def register_handler(self, category: str, handler: PurgeHandler) -> None:
        """Register a purge handler for a data category."""
        self._handlers[category] = handler

    def unregister_handler(self, category: str) -> bool:
        """Unregister a handler.  Returns True if it existed."""
        return self._handlers.pop(category, None) is not None

    # -- enforcement --------------------------------------------------------

    def enforce(self, now: Optional[float] = None) -> list[PurgeResult]:
        """Enforce all active policies.

        For each enabled policy that has a registered handler, compute
        the cutoff timestamp and invoke the handler.

        Returns a list of :class:`PurgeResult` for each policy enforced.
        """
        now = now if now is not None else time.time()
        results: list[PurgeResult] = []

        for category, policy in sorted(self._policies.items()):
            if not policy.enabled:
                continue
            handler = self._handlers.get(category)
            if handler is None:
                continue

            cutoff = now - policy.retention_seconds
            t0 = time.monotonic()
            try:
                count = handler(category, cutoff)
                elapsed = time.monotonic() - t0
                result = PurgeResult(
                    category=category,
                    purged_count=count,
                    cutoff_timestamp=cutoff,
                    execution_time=elapsed,
                )
            except Exception as exc:
                elapsed = time.monotonic() - t0
                result = PurgeResult(
                    category=category,
                    purged_count=0,
                    cutoff_timestamp=cutoff,
                    execution_time=elapsed,
                    errors=[str(exc)],
                )
                logger.error("Purge failed for %s: %s", category, exc)

            results.append(result)
            self._history.append(result)

        return results

    def enforce_category(
        self, category: str, now: Optional[float] = None
    ) -> Optional[PurgeResult]:
        """Enforce a single category.  Returns None if no policy or handler."""
        policy = self._policies.get(category)
        if policy is None or not policy.enabled:
            return None
        handler = self._handlers.get(category)
        if handler is None:
            return None

        now = now if now is not None else time.time()
        cutoff = now - policy.retention_seconds
        t0 = time.monotonic()
        try:
            count = handler(category, cutoff)
            elapsed = time.monotonic() - t0
            result = PurgeResult(
                category=category,
                purged_count=count,
                cutoff_timestamp=cutoff,
                execution_time=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            result = PurgeResult(
                category=category,
                purged_count=0,
                cutoff_timestamp=cutoff,
                execution_time=elapsed,
                errors=[str(exc)],
            )
        self._history.append(result)
        return result

    # -- history ------------------------------------------------------------

    @property
    def history(self) -> list[PurgeResult]:
        """Return all past enforcement results."""
        return list(self._history)

    def clear_history(self) -> int:
        """Clear enforcement history.  Returns count cleared."""
        count = len(self._history)
        self._history.clear()
        return count

    # -- export -------------------------------------------------------------

    def export(self) -> dict[str, Any]:
        """Export full manager state as a serializable dict."""
        return {
            "policies": {k: v.to_dict() for k, v in self._policies.items()},
            "handlers_registered": sorted(self._handlers.keys()),
            "history_count": len(self._history),
        }
