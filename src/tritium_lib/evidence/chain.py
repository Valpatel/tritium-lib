# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Chain of custody tracking for evidence items.

Records every access, transfer, and modification of evidence so that
a complete audit trail exists from collection through presentation.
Each custody event is timestamped and attributed to an actor.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class CustodyAction(str, Enum):
    """Actions that can occur in the chain of custody."""
    COLLECTED = "collected"          # Initial collection
    ACCESSED = "accessed"            # Read/viewed
    TRANSFERRED = "transferred"      # Moved between custodians
    VERIFIED = "verified"            # Integrity check passed
    SEALED = "sealed"                # Locked against modification
    CHALLENGED = "challenged"        # Integrity check failed
    ANNOTATED = "annotated"          # Notes/tags added
    EXPORTED = "exported"            # Included in an export package
    ARCHIVED = "archived"            # Moved to long-term storage
    EXPUNGED = "expunged"            # Marked for deletion


class CustodyEvent(BaseModel):
    """A single event in the chain of custody.

    Attributes:
        event_id: Unique identifier for this custody event.
        evidence_id: Which evidence item this event pertains to.
        action: What happened.
        actor: Who performed the action (user ID, system name).
        timestamp: When the action occurred (UTC).
        from_custodian: Previous custodian (for transfers).
        to_custodian: New custodian (for transfers).
        details: Additional context about the action.
        ip_address: IP of the accessor (when available).
        sha256_at_time: Hash of the evidence at the time of this event.
    """
    event_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:16],
        description="Unique identifier for this custody event",
    )
    evidence_id: str = Field(
        ...,
        description="Evidence item this event pertains to",
    )
    action: CustodyAction = Field(
        ...,
        description="What happened",
    )
    actor: str = Field(
        "",
        description="Who performed the action",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the action occurred (UTC)",
    )
    from_custodian: str = Field(
        "",
        description="Previous custodian (for transfers)",
    )
    to_custodian: str = Field(
        "",
        description="New custodian (for transfers)",
    )
    details: str = Field(
        "",
        description="Additional context about the action",
    )
    ip_address: str = Field(
        "",
        description="IP address of the accessor",
    )
    sha256_at_time: str = Field(
        "",
        description="SHA-256 hash of the evidence at the time of this event",
    )

    model_config = {"frozen": False}


class EvidenceChain(BaseModel):
    """Chain of custody for a single evidence item.

    Maintains an ordered list of custody events from collection
    through all subsequent accesses, transfers, and verifications.

    Attributes:
        evidence_id: The evidence item this chain tracks.
        custodian: Current custodian of the evidence.
        events: Ordered list of custody events.
        created_at: When the chain was created.
    """
    evidence_id: str = Field(
        ...,
        description="Evidence item this chain tracks",
    )
    custodian: str = Field(
        "",
        description="Current custodian of the evidence",
    )
    events: list[CustodyEvent] = Field(
        default_factory=list,
        description="Ordered list of custody events",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the chain was created",
    )

    model_config = {"frozen": False}

    def record_collection(
        self,
        actor: str,
        details: str = "",
        sha256: str = "",
    ) -> CustodyEvent:
        """Record the initial collection of evidence.

        Args:
            actor: Who collected the evidence.
            details: Additional context.
            sha256: Hash of the evidence at collection time.

        Returns:
            The created CustodyEvent.
        """
        event = CustodyEvent(
            evidence_id=self.evidence_id,
            action=CustodyAction.COLLECTED,
            actor=actor,
            to_custodian=actor,
            details=details,
            sha256_at_time=sha256,
        )
        self.events.append(event)
        self.custodian = actor
        return event

    def record_access(
        self,
        actor: str,
        details: str = "",
        ip_address: str = "",
    ) -> CustodyEvent:
        """Record that someone accessed/viewed the evidence.

        Args:
            actor: Who accessed the evidence.
            details: What they did with it.
            ip_address: IP address of the accessor.

        Returns:
            The created CustodyEvent.
        """
        event = CustodyEvent(
            evidence_id=self.evidence_id,
            action=CustodyAction.ACCESSED,
            actor=actor,
            details=details,
            ip_address=ip_address,
        )
        self.events.append(event)
        return event

    def record_transfer(
        self,
        from_custodian: str,
        to_custodian: str,
        actor: str = "",
        details: str = "",
        sha256: str = "",
    ) -> CustodyEvent:
        """Record a transfer of custody.

        Args:
            from_custodian: Previous custodian.
            to_custodian: New custodian.
            actor: Who authorized the transfer (defaults to from_custodian).
            details: Reason for transfer.
            sha256: Hash at time of transfer.

        Returns:
            The created CustodyEvent.
        """
        event = CustodyEvent(
            evidence_id=self.evidence_id,
            action=CustodyAction.TRANSFERRED,
            actor=actor or from_custodian,
            from_custodian=from_custodian,
            to_custodian=to_custodian,
            details=details,
            sha256_at_time=sha256,
        )
        self.events.append(event)
        self.custodian = to_custodian
        return event

    def record_verification(
        self,
        actor: str,
        passed: bool,
        sha256: str = "",
        details: str = "",
    ) -> CustodyEvent:
        """Record an integrity verification.

        Args:
            actor: Who performed the verification.
            passed: Whether the integrity check passed.
            sha256: Hash at time of verification.
            details: Additional context.

        Returns:
            The created CustodyEvent.
        """
        action = CustodyAction.VERIFIED if passed else CustodyAction.CHALLENGED
        event = CustodyEvent(
            evidence_id=self.evidence_id,
            action=action,
            actor=actor,
            details=details or f"Integrity {'PASSED' if passed else 'FAILED'}",
            sha256_at_time=sha256,
        )
        self.events.append(event)
        return event

    def record_seal(self, actor: str, sha256: str = "") -> CustodyEvent:
        """Record that evidence was sealed against modification.

        Args:
            actor: Who sealed the evidence.
            sha256: Hash at time of sealing.

        Returns:
            The created CustodyEvent.
        """
        event = CustodyEvent(
            evidence_id=self.evidence_id,
            action=CustodyAction.SEALED,
            actor=actor,
            details="Evidence sealed — no further modifications permitted",
            sha256_at_time=sha256,
        )
        self.events.append(event)
        return event

    def record_export(self, actor: str, details: str = "") -> CustodyEvent:
        """Record that evidence was exported.

        Args:
            actor: Who exported the evidence.
            details: Export destination or format.

        Returns:
            The created CustodyEvent.
        """
        event = CustodyEvent(
            evidence_id=self.evidence_id,
            action=CustodyAction.EXPORTED,
            actor=actor,
            details=details,
        )
        self.events.append(event)
        return event

    def record_annotation(self, actor: str, details: str) -> CustodyEvent:
        """Record that evidence was annotated.

        Args:
            actor: Who annotated the evidence.
            details: What was added.

        Returns:
            The created CustodyEvent.
        """
        event = CustodyEvent(
            evidence_id=self.evidence_id,
            action=CustodyAction.ANNOTATED,
            actor=actor,
            details=details,
        )
        self.events.append(event)
        return event

    @property
    def event_count(self) -> int:
        """Total number of custody events."""
        return len(self.events)

    @property
    def last_event(self) -> Optional[CustodyEvent]:
        """Most recent custody event, or None."""
        return self.events[-1] if self.events else None

    def get_events_by_action(self, action: CustodyAction) -> list[CustodyEvent]:
        """Filter events by action type.

        Args:
            action: Action type to filter by.

        Returns:
            List of matching events.
        """
        return [e for e in self.events if e.action == action]

    def get_actors(self) -> list[str]:
        """Return a deduplicated list of actors who touched this evidence.

        Returns:
            List of unique actor identifiers.
        """
        seen: set[str] = set()
        actors: list[str] = []
        for e in self.events:
            if e.actor and e.actor not in seen:
                seen.add(e.actor)
                actors.append(e.actor)
        return actors

    def to_summary(self) -> dict[str, Any]:
        """Return a concise summary dict for display."""
        return {
            "evidence_id": self.evidence_id,
            "custodian": self.custodian,
            "event_count": self.event_count,
            "actors": self.get_actors(),
            "created_at": self.created_at.isoformat(),
            "last_action": self.last_event.action.value if self.last_event else None,
        }
