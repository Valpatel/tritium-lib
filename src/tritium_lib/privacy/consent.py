# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Consent management for data processing.

Tracks what data processing a data subject has consented to,
when consent was given/withdrawn, and the legal basis for processing.

Under GDPR Article 7, consent must be:
    - Freely given, specific, informed, and unambiguous
    - Demonstrable (we must prove it was given)
    - Withdrawable at any time

This module provides the bookkeeping; the UI/UX for collecting consent
lives in the command center frontend.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("privacy.consent")


# ---------------------------------------------------------------------------
# Processing purposes
# ---------------------------------------------------------------------------

class ProcessingPurpose(str, Enum):
    """Purposes for which data may be processed."""
    TRACKING = "tracking"
    IDENTIFICATION = "identification"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    LOCATION_HISTORY = "location_history"
    CAMERA_MONITORING = "camera_monitoring"
    THREAT_ASSESSMENT = "threat_assessment"
    DOSSIER_BUILDING = "dossier_building"
    SENSOR_FUSION = "sensor_fusion"
    ANALYTICS = "analytics"
    TRAINING_DATA = "training_data"


class LegalBasis(str, Enum):
    """GDPR legal bases for processing (Article 6)."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_INTEREST = "public_interest"
    LEGITIMATE_INTEREST = "legitimate_interest"


class ConsentStatus(str, Enum):
    """Current state of a consent record."""
    GRANTED = "granted"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    PENDING = "pending"


# ---------------------------------------------------------------------------
# ConsentRecord
# ---------------------------------------------------------------------------

@dataclass
class ConsentRecord:
    """A single consent decision by a data subject.

    Attributes
    ----------
    consent_id : str
        Unique identifier for this consent record.
    subject_id : str
        Identifier for the data subject (e.g. target ID, user ID).
    purpose : str
        Processing purpose this consent covers.
    status : str
        Current consent status.
    legal_basis : str
        Legal basis for processing.
    granted_at : float
        Unix timestamp when consent was granted (0 if pending).
    withdrawn_at : float
        Unix timestamp when consent was withdrawn (0 if still active).
    expires_at : float
        Unix timestamp when consent expires (0 = no expiry).
    evidence : str
        How consent was obtained (e.g. "web_form", "api", "verbal").
    notes : str
        Additional context.
    """

    consent_id: str = ""
    subject_id: str = ""
    purpose: str = ""
    status: str = "pending"
    legal_basis: str = "consent"
    granted_at: float = 0.0
    withdrawn_at: float = 0.0
    expires_at: float = 0.0
    evidence: str = ""
    notes: str = ""

    def is_active(self, now: Optional[float] = None) -> bool:
        """Return True if this consent is currently valid."""
        if self.status != ConsentStatus.GRANTED:
            return False
        now = now if now is not None else time.time()
        if self.expires_at > 0 and now > self.expires_at:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "consent_id": self.consent_id,
            "subject_id": self.subject_id,
            "purpose": self.purpose,
            "status": self.status,
            "legal_basis": self.legal_basis,
            "granted_at": self.granted_at,
            "withdrawn_at": self.withdrawn_at,
            "expires_at": self.expires_at,
            "evidence": self.evidence,
            "notes": self.notes,
        }

    @staticmethod
    def create(
        subject_id: str,
        purpose: str,
        legal_basis: str = "consent",
        evidence: str = "",
        expires_at: float = 0.0,
        notes: str = "",
    ) -> ConsentRecord:
        """Factory: create a new granted consent record."""
        return ConsentRecord(
            consent_id=str(uuid.uuid4()),
            subject_id=subject_id,
            purpose=purpose,
            status=ConsentStatus.GRANTED,
            legal_basis=legal_basis,
            granted_at=time.time(),
            expires_at=expires_at,
            evidence=evidence,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# ConsentManager
# ---------------------------------------------------------------------------

class ConsentManager:
    """Track and enforce data processing consent.

    Maintains an in-memory registry of consent records.  For production
    use, hook up persistence via ``export()`` / ``import_records()``.

    Usage
    -----
    ::

        mgr = ConsentManager()
        record = mgr.grant("target_123", "tracking", evidence="web_form")
        assert mgr.has_consent("target_123", "tracking")
        mgr.withdraw("target_123", "tracking")
        assert not mgr.has_consent("target_123", "tracking")
    """

    def __init__(self) -> None:
        # {subject_id -> {purpose -> ConsentRecord}}
        self._records: dict[str, dict[str, ConsentRecord]] = {}
        self._history: list[ConsentRecord] = []

    # -- grant / withdraw ---------------------------------------------------

    def grant(
        self,
        subject_id: str,
        purpose: str,
        legal_basis: str = "consent",
        evidence: str = "",
        expires_at: float = 0.0,
        notes: str = "",
    ) -> ConsentRecord:
        """Record that a subject has granted consent for a purpose."""
        record = ConsentRecord.create(
            subject_id=subject_id,
            purpose=purpose,
            legal_basis=legal_basis,
            evidence=evidence,
            expires_at=expires_at,
            notes=notes,
        )
        self._records.setdefault(subject_id, {})[purpose] = record
        self._history.append(record)
        logger.info("Consent granted: %s -> %s", subject_id, purpose)
        return record

    def withdraw(self, subject_id: str, purpose: str) -> Optional[ConsentRecord]:
        """Withdraw consent for a specific purpose.

        Returns the updated record, or None if no consent existed.
        """
        subject_records = self._records.get(subject_id, {})
        record = subject_records.get(purpose)
        if record is None:
            return None

        updated = ConsentRecord(
            consent_id=record.consent_id,
            subject_id=record.subject_id,
            purpose=record.purpose,
            status=ConsentStatus.WITHDRAWN,
            legal_basis=record.legal_basis,
            granted_at=record.granted_at,
            withdrawn_at=time.time(),
            expires_at=record.expires_at,
            evidence=record.evidence,
            notes=record.notes,
        )
        subject_records[purpose] = updated
        self._history.append(updated)
        logger.info("Consent withdrawn: %s -> %s", subject_id, purpose)
        return updated

    def withdraw_all(self, subject_id: str) -> list[ConsentRecord]:
        """Withdraw all consents for a subject.  Returns updated records."""
        subject_records = self._records.get(subject_id, {})
        results = []
        for purpose in list(subject_records.keys()):
            result = self.withdraw(subject_id, purpose)
            if result is not None:
                results.append(result)
        return results

    # -- queries ------------------------------------------------------------

    def has_consent(
        self,
        subject_id: str,
        purpose: str,
        now: Optional[float] = None,
    ) -> bool:
        """Check if a subject has active consent for a purpose."""
        record = self._records.get(subject_id, {}).get(purpose)
        if record is None:
            return False
        return record.is_active(now=now)

    def get_consent(
        self, subject_id: str, purpose: str
    ) -> Optional[ConsentRecord]:
        """Get the consent record for a subject and purpose."""
        return self._records.get(subject_id, {}).get(purpose)

    def get_all_consents(self, subject_id: str) -> list[ConsentRecord]:
        """Get all consent records for a subject."""
        return list(self._records.get(subject_id, {}).values())

    def get_active_consents(
        self, subject_id: str, now: Optional[float] = None
    ) -> list[ConsentRecord]:
        """Get only active (granted, non-expired) consents for a subject."""
        return [
            r for r in self.get_all_consents(subject_id)
            if r.is_active(now=now)
        ]

    def list_subjects(self) -> list[str]:
        """List all subject IDs with consent records."""
        return sorted(self._records.keys())

    def count_by_purpose(self) -> dict[str, int]:
        """Count active consents per purpose."""
        counts: dict[str, int] = {}
        for subject_records in self._records.values():
            for purpose, record in subject_records.items():
                if record.is_active():
                    counts[purpose] = counts.get(purpose, 0) + 1
        return counts

    # -- history / export ---------------------------------------------------

    @property
    def history(self) -> list[ConsentRecord]:
        """Return the full consent audit trail."""
        return list(self._history)

    def export(self) -> dict[str, Any]:
        """Export all records for serialization/persistence."""
        return {
            "records": {
                sid: {p: r.to_dict() for p, r in recs.items()}
                for sid, recs in self._records.items()
            },
            "history_count": len(self._history),
            "subject_count": len(self._records),
        }

    def import_records(self, records: list[ConsentRecord]) -> int:
        """Import consent records.  Returns count imported."""
        count = 0
        for record in records:
            self._records.setdefault(record.subject_id, {})[record.purpose] = record
            count += 1
        return count

    def clear(self) -> None:
        """Clear all records and history."""
        self._records.clear()
        self._history.clear()
