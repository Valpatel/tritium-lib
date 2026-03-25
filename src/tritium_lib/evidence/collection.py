# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Evidence collection — grouped evidence for an investigation.

An EvidenceCollection aggregates multiple Evidence items under a
single investigation, maintaining chains of custody for each item
and providing query/filter capabilities.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from .chain import CustodyAction, EvidenceChain
from .integrity import hash_evidence, verify_integrity
from .models import Evidence, EvidenceStatus, EvidenceType


class InvestigationStatus(str, Enum):
    """Status of an investigation."""
    OPEN = "open"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"
    ARCHIVED = "archived"


class EvidenceCollection(BaseModel):
    """Grouped evidence for an investigation.

    Manages a set of Evidence items with their chain-of-custody records.
    Provides methods for adding, searching, verifying, and sealing
    evidence within the scope of a single investigation.

    Attributes:
        collection_id: Unique identifier for this collection.
        title: Human-readable title of the investigation.
        description: Detailed description of the investigation scope.
        status: Current investigation status.
        created_at: When the investigation was opened.
        created_by: Who opened the investigation.
        target_ids: Target IDs that are subjects of this investigation.
        evidence: Dictionary of evidence items keyed by evidence_id.
        chains: Dictionary of custody chains keyed by evidence_id.
        tags: Free-form tags for categorization.
    """
    collection_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for this collection",
    )
    title: str = Field(
        "",
        description="Human-readable title of the investigation",
    )
    description: str = Field(
        "",
        description="Detailed description of the investigation scope",
    )
    status: InvestigationStatus = Field(
        default=InvestigationStatus.OPEN,
        description="Current investigation status",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the investigation was opened",
    )
    created_by: str = Field(
        "",
        description="Who opened the investigation",
    )
    target_ids: list[str] = Field(
        default_factory=list,
        description="Target IDs that are subjects of this investigation",
    )
    evidence: dict[str, Evidence] = Field(
        default_factory=dict,
        description="Evidence items keyed by evidence_id",
    )
    chains: dict[str, EvidenceChain] = Field(
        default_factory=dict,
        description="Custody chains keyed by evidence_id",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags",
    )

    model_config = {"frozen": False}

    def add_evidence(
        self,
        evidence: Evidence,
        collector: str = "",
    ) -> Evidence:
        """Add an evidence item to the collection.

        Computes the integrity hash, creates a chain-of-custody record,
        and links the evidence to this collection.

        Args:
            evidence: Evidence item to add.
            collector: Who is adding this evidence.

        Returns:
            The evidence item (with hash and investigation_id set).
        """
        evidence.investigation_id = self.collection_id
        hash_evidence(evidence)

        # Create chain of custody
        chain = EvidenceChain(evidence_id=evidence.evidence_id)
        chain.record_collection(
            actor=collector or self.created_by or "system",
            sha256=evidence.sha256,
        )

        self.evidence[evidence.evidence_id] = evidence
        self.chains[evidence.evidence_id] = chain
        return evidence

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        """Retrieve an evidence item by ID.

        Args:
            evidence_id: The evidence item's unique identifier.

        Returns:
            The Evidence item, or None if not found.
        """
        return self.evidence.get(evidence_id)

    def get_chain(self, evidence_id: str) -> Optional[EvidenceChain]:
        """Retrieve the custody chain for an evidence item.

        Args:
            evidence_id: The evidence item's unique identifier.

        Returns:
            The EvidenceChain, or None if not found.
        """
        return self.chains.get(evidence_id)

    def remove_evidence(self, evidence_id: str) -> bool:
        """Remove an evidence item from the collection.

        Only works if the evidence is not SEALED.

        Args:
            evidence_id: The evidence item to remove.

        Returns:
            True if removed, False if not found or sealed.
        """
        ev = self.evidence.get(evidence_id)
        if ev is None:
            return False
        if ev.status == EvidenceStatus.SEALED:
            return False
        del self.evidence[evidence_id]
        self.chains.pop(evidence_id, None)
        return True

    def find_by_type(self, evidence_type: EvidenceType) -> list[Evidence]:
        """Find all evidence items of a specific type.

        Args:
            evidence_type: Type to filter by.

        Returns:
            List of matching evidence items.
        """
        return [
            e for e in self.evidence.values()
            if e.evidence_type == evidence_type
        ]

    def find_by_target(self, target_id: str) -> list[Evidence]:
        """Find all evidence items related to a specific target.

        Args:
            target_id: Target ID to filter by.

        Returns:
            List of matching evidence items.
        """
        return [
            e for e in self.evidence.values()
            if e.target_id == target_id
        ]

    def find_by_tag(self, tag: str) -> list[Evidence]:
        """Find all evidence items with a specific tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List of matching evidence items.
        """
        return [
            e for e in self.evidence.values()
            if tag in e.tags
        ]

    def find_by_status(self, status: EvidenceStatus) -> list[Evidence]:
        """Find all evidence items with a given lifecycle status.

        Args:
            status: Status to filter by.

        Returns:
            List of matching evidence items.
        """
        return [
            e for e in self.evidence.values()
            if e.status == status
        ]

    def verify_all(self) -> dict[str, bool]:
        """Verify integrity of all evidence items.

        Returns:
            Dictionary mapping evidence_id to verification result.
        """
        results: dict[str, bool] = {}
        for eid, ev in self.evidence.items():
            passed = verify_integrity(ev)
            results[eid] = passed
            chain = self.chains.get(eid)
            if chain:
                chain.record_verification(
                    actor="system",
                    passed=passed,
                    sha256=ev.sha256,
                )
        return results

    def seal_all(self, actor: str = "system") -> int:
        """Seal all evidence items in the collection.

        Args:
            actor: Who is sealing the evidence.

        Returns:
            Number of items sealed.
        """
        count = 0
        for eid, ev in self.evidence.items():
            if ev.status != EvidenceStatus.SEALED:
                ev.seal()
                chain = self.chains.get(eid)
                if chain:
                    chain.record_seal(actor=actor, sha256=ev.sha256)
                count += 1
        return count

    def close(self) -> None:
        """Close the investigation."""
        self.status = InvestigationStatus.CLOSED

    def archive(self) -> None:
        """Archive the investigation."""
        self.status = InvestigationStatus.ARCHIVED

    @property
    def evidence_count(self) -> int:
        """Total number of evidence items."""
        return len(self.evidence)

    @property
    def target_count(self) -> int:
        """Number of unique targets referenced in evidence."""
        return len({e.target_id for e in self.evidence.values() if e.target_id})

    def get_type_counts(self) -> dict[str, int]:
        """Count evidence items by type.

        Returns:
            Dictionary mapping evidence type name to count.
        """
        counts: dict[str, int] = {}
        for ev in self.evidence.values():
            key = ev.evidence_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_manifest(self) -> dict[str, Any]:
        """Generate a manifest dict describing the collection.

        Returns:
            Manifest dictionary suitable for JSON serialization.
        """
        return {
            "collection_id": self.collection_id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "target_ids": self.target_ids,
            "evidence_count": self.evidence_count,
            "type_counts": self.get_type_counts(),
            "tags": self.tags,
            "items": [
                {
                    "evidence_id": ev.evidence_id,
                    "type": ev.evidence_type.value,
                    "target_id": ev.target_id,
                    "status": ev.status.value,
                    "sha256": ev.sha256,
                    "collected_at": ev.collected_at.isoformat(),
                    "collected_by": ev.collected_by,
                    "custody_events": (
                        self.chains[ev.evidence_id].event_count
                        if ev.evidence_id in self.chains else 0
                    ),
                }
                for ev in self.evidence.values()
            ],
        }
