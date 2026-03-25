# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""GDPR-style data subject access and deletion requests.

Implements the rights of data subjects under GDPR:
    - Right of access (Article 15): export all data about a subject
    - Right to erasure (Article 17): delete all data about a subject
    - Right to rectification (Article 16): correct inaccurate data
    - Right to portability (Article 20): export data in machine-readable format
    - Right to restriction (Article 18): halt processing while disputes are resolved

Each request goes through a lifecycle:
    PENDING -> PROCESSING -> COMPLETED / DENIED / FAILED
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger("privacy.subject_request")


# ---------------------------------------------------------------------------
# Request types and statuses
# ---------------------------------------------------------------------------

class RequestType(str, Enum):
    """Types of data subject requests (GDPR articles)."""
    ACCESS = "access"              # Article 15 — give me my data
    ERASURE = "erasure"            # Article 17 — delete my data
    RECTIFICATION = "rectification"  # Article 16 — fix my data
    PORTABILITY = "portability"    # Article 20 — export my data
    RESTRICTION = "restriction"    # Article 18 — stop processing


class RequestStatus(str, Enum):
    """Lifecycle status of a subject request."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    DENIED = "denied"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# DataSubjectRequest
# ---------------------------------------------------------------------------

@dataclass
class DataSubjectRequest:
    """A request from a data subject to exercise their privacy rights.

    Attributes
    ----------
    request_id : str
        Unique identifier for this request.
    subject_id : str
        Who is making the request (target ID, user ID, etc.).
    request_type : str
        What right is being exercised.
    status : str
        Current lifecycle status.
    reason : str
        Why the request is being made (free text).
    created_at : float
        When the request was submitted.
    updated_at : float
        When the request was last updated.
    completed_at : float
        When the request was fulfilled (0 if not yet completed).
    response_data : dict
        Data returned to the subject (for access/portability requests).
    affected_records : int
        Number of records affected (for erasure/rectification).
    processor : str
        Who/what is handling this request.
    denial_reason : str
        If denied, why.
    notes : str
        Internal notes about the request.
    """

    request_id: str = ""
    subject_id: str = ""
    request_type: str = ""
    status: str = "pending"
    reason: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    completed_at: float = 0.0
    response_data: dict = field(default_factory=dict)
    affected_records: int = 0
    processor: str = ""
    denial_reason: str = ""
    notes: str = ""

    @property
    def is_open(self) -> bool:
        """True if the request is not yet resolved."""
        return self.status in (RequestStatus.PENDING, RequestStatus.PROCESSING)

    @property
    def response_time_seconds(self) -> Optional[float]:
        """Time from creation to completion, or None if still open."""
        if self.completed_at > 0:
            return self.completed_at - self.created_at
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "subject_id": self.subject_id,
            "request_type": self.request_type,
            "status": self.status,
            "reason": self.reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "response_data": dict(self.response_data),
            "affected_records": self.affected_records,
            "processor": self.processor,
            "denial_reason": self.denial_reason,
            "notes": self.notes,
        }

    @staticmethod
    def create(
        subject_id: str,
        request_type: str,
        reason: str = "",
        notes: str = "",
    ) -> DataSubjectRequest:
        """Factory: create a new pending request."""
        now = time.time()
        return DataSubjectRequest(
            request_id=str(uuid.uuid4()),
            subject_id=subject_id,
            request_type=request_type,
            status=RequestStatus.PENDING,
            reason=reason,
            created_at=now,
            updated_at=now,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Handler type for request fulfillment
# ---------------------------------------------------------------------------

SubjectDataCollector = Callable[[str], dict[str, Any]]
"""Callable(subject_id) -> dict of all data for that subject."""

SubjectDataEraser = Callable[[str], int]
"""Callable(subject_id) -> count of records deleted."""


# ---------------------------------------------------------------------------
# SubjectRequestManager
# ---------------------------------------------------------------------------

class SubjectRequestManager:
    """Process GDPR-style data subject requests.

    Register collectors and erasers for different data stores, then
    submit and process requests.

    Usage
    -----
    ::

        mgr = SubjectRequestManager()
        mgr.register_collector("sightings", get_sightings)
        mgr.register_eraser("sightings", delete_sightings)

        req = mgr.submit_access("target_123", reason="Subject request")
        mgr.process(req.request_id)
        # req is now COMPLETED with response_data populated

    GDPR requires responses within 30 days.  The ``overdue()`` method
    helps track requests approaching the deadline.
    """

    # 30-day GDPR response deadline
    RESPONSE_DEADLINE_SECONDS = 30 * 86400

    def __init__(self) -> None:
        self._requests: dict[str, DataSubjectRequest] = {}
        self._collectors: dict[str, SubjectDataCollector] = {}
        self._erasers: dict[str, SubjectDataEraser] = {}

    # -- submit requests ----------------------------------------------------

    def submit_access(
        self, subject_id: str, reason: str = ""
    ) -> DataSubjectRequest:
        """Submit a right-of-access request."""
        req = DataSubjectRequest.create(
            subject_id=subject_id,
            request_type=RequestType.ACCESS,
            reason=reason,
        )
        self._requests[req.request_id] = req
        logger.info("Access request submitted: %s for %s",
                     req.request_id, subject_id)
        return req

    def submit_erasure(
        self, subject_id: str, reason: str = ""
    ) -> DataSubjectRequest:
        """Submit a right-to-erasure request."""
        req = DataSubjectRequest.create(
            subject_id=subject_id,
            request_type=RequestType.ERASURE,
            reason=reason,
        )
        self._requests[req.request_id] = req
        logger.info("Erasure request submitted: %s for %s",
                     req.request_id, subject_id)
        return req

    def submit_portability(
        self, subject_id: str, reason: str = ""
    ) -> DataSubjectRequest:
        """Submit a data portability request."""
        req = DataSubjectRequest.create(
            subject_id=subject_id,
            request_type=RequestType.PORTABILITY,
            reason=reason,
        )
        self._requests[req.request_id] = req
        return req

    def submit_rectification(
        self, subject_id: str, reason: str = ""
    ) -> DataSubjectRequest:
        """Submit a rectification request."""
        req = DataSubjectRequest.create(
            subject_id=subject_id,
            request_type=RequestType.RECTIFICATION,
            reason=reason,
        )
        self._requests[req.request_id] = req
        return req

    def submit_restriction(
        self, subject_id: str, reason: str = ""
    ) -> DataSubjectRequest:
        """Submit a processing restriction request."""
        req = DataSubjectRequest.create(
            subject_id=subject_id,
            request_type=RequestType.RESTRICTION,
            reason=reason,
        )
        self._requests[req.request_id] = req
        return req

    # -- register handlers --------------------------------------------------

    def register_collector(self, store_name: str, collector: SubjectDataCollector) -> None:
        """Register a data collector for a named data store."""
        self._collectors[store_name] = collector

    def register_eraser(self, store_name: str, eraser: SubjectDataEraser) -> None:
        """Register a data eraser for a named data store."""
        self._erasers[store_name] = eraser

    # -- process requests ---------------------------------------------------

    def process(self, request_id: str) -> DataSubjectRequest:
        """Process a pending request.

        For ACCESS/PORTABILITY: collects data from all registered collectors.
        For ERASURE: deletes data from all registered erasers.
        For RECTIFICATION/RESTRICTION: marked as processing (manual step needed).

        Returns the updated request.

        Raises
        ------
        KeyError
            If request_id is not found.
        ValueError
            If the request is not in a processable state.
        """
        req = self._requests.get(request_id)
        if req is None:
            raise KeyError(f"Request not found: {request_id}")
        if not req.is_open:
            raise ValueError(
                f"Request {request_id} is already {req.status}, cannot process"
            )

        req = DataSubjectRequest(
            request_id=req.request_id,
            subject_id=req.subject_id,
            request_type=req.request_type,
            status=RequestStatus.PROCESSING,
            reason=req.reason,
            created_at=req.created_at,
            updated_at=time.time(),
            completed_at=req.completed_at,
            response_data=dict(req.response_data),
            affected_records=req.affected_records,
            processor=req.processor,
            denial_reason=req.denial_reason,
            notes=req.notes,
        )
        self._requests[request_id] = req

        rtype = req.request_type

        if rtype in (RequestType.ACCESS, RequestType.PORTABILITY):
            return self._process_access(req)
        elif rtype == RequestType.ERASURE:
            return self._process_erasure(req)
        else:
            # RECTIFICATION and RESTRICTION need manual intervention
            # Leave in PROCESSING status
            self._requests[request_id] = req
            return req

    def complete(
        self,
        request_id: str,
        notes: str = "",
        affected_records: int = 0,
    ) -> DataSubjectRequest:
        """Manually complete a request (for rectification/restriction)."""
        req = self._requests.get(request_id)
        if req is None:
            raise KeyError(f"Request not found: {request_id}")

        now = time.time()
        updated = DataSubjectRequest(
            request_id=req.request_id,
            subject_id=req.subject_id,
            request_type=req.request_type,
            status=RequestStatus.COMPLETED,
            reason=req.reason,
            created_at=req.created_at,
            updated_at=now,
            completed_at=now,
            response_data=dict(req.response_data),
            affected_records=affected_records or req.affected_records,
            processor=req.processor,
            denial_reason=req.denial_reason,
            notes=notes or req.notes,
        )
        self._requests[request_id] = updated
        return updated

    def deny(
        self, request_id: str, denial_reason: str = ""
    ) -> DataSubjectRequest:
        """Deny a request with a reason."""
        req = self._requests.get(request_id)
        if req is None:
            raise KeyError(f"Request not found: {request_id}")

        now = time.time()
        updated = DataSubjectRequest(
            request_id=req.request_id,
            subject_id=req.subject_id,
            request_type=req.request_type,
            status=RequestStatus.DENIED,
            reason=req.reason,
            created_at=req.created_at,
            updated_at=now,
            completed_at=now,
            response_data=dict(req.response_data),
            affected_records=req.affected_records,
            processor=req.processor,
            denial_reason=denial_reason,
            notes=req.notes,
        )
        self._requests[request_id] = updated
        return updated

    # -- queries ------------------------------------------------------------

    def get(self, request_id: str) -> Optional[DataSubjectRequest]:
        """Get a request by ID."""
        return self._requests.get(request_id)

    def list_requests(
        self,
        subject_id: Optional[str] = None,
        status: Optional[str] = None,
        request_type: Optional[str] = None,
    ) -> list[DataSubjectRequest]:
        """List requests with optional filters."""
        results = list(self._requests.values())
        if subject_id is not None:
            results = [r for r in results if r.subject_id == subject_id]
        if status is not None:
            results = [r for r in results if r.status == status]
        if request_type is not None:
            results = [r for r in results if r.request_type == request_type]
        return sorted(results, key=lambda r: r.created_at, reverse=True)

    def overdue(self, now: Optional[float] = None) -> list[DataSubjectRequest]:
        """Return open requests past the 30-day GDPR deadline."""
        now = now if now is not None else time.time()
        deadline = now - self.RESPONSE_DEADLINE_SECONDS
        return [
            r for r in self._requests.values()
            if r.is_open and r.created_at < deadline
        ]

    def stats(self) -> dict[str, Any]:
        """Summary statistics."""
        by_status: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for r in self._requests.values():
            by_status[r.status] = by_status.get(r.status, 0) + 1
            by_type[r.request_type] = by_type.get(r.request_type, 0) + 1
        return {
            "total": len(self._requests),
            "by_status": by_status,
            "by_type": by_type,
            "overdue": len(self.overdue()),
        }

    # -- internals ----------------------------------------------------------

    def _process_access(self, req: DataSubjectRequest) -> DataSubjectRequest:
        """Collect data from all registered collectors."""
        all_data: dict[str, Any] = {}
        total_records = 0

        for store_name, collector in sorted(self._collectors.items()):
            try:
                data = collector(req.subject_id)
                all_data[store_name] = data
                if isinstance(data, dict):
                    total_records += len(data)
                elif isinstance(data, list):
                    total_records += len(data)
            except Exception as exc:
                all_data[store_name] = {"error": str(exc)}
                logger.error("Collector %s failed for %s: %s",
                             store_name, req.subject_id, exc)

        now = time.time()
        updated = DataSubjectRequest(
            request_id=req.request_id,
            subject_id=req.subject_id,
            request_type=req.request_type,
            status=RequestStatus.COMPLETED,
            reason=req.reason,
            created_at=req.created_at,
            updated_at=now,
            completed_at=now,
            response_data=all_data,
            affected_records=total_records,
            processor="auto",
            denial_reason=req.denial_reason,
            notes=req.notes,
        )
        self._requests[req.request_id] = updated
        return updated

    def _process_erasure(self, req: DataSubjectRequest) -> DataSubjectRequest:
        """Delete data from all registered erasers."""
        total_deleted = 0
        errors: list[str] = []

        for store_name, eraser in sorted(self._erasers.items()):
            try:
                count = eraser(req.subject_id)
                total_deleted += count
            except Exception as exc:
                errors.append(f"{store_name}: {exc}")
                logger.error("Eraser %s failed for %s: %s",
                             store_name, req.subject_id, exc)

        now = time.time()
        status = RequestStatus.COMPLETED if not errors else RequestStatus.FAILED
        updated = DataSubjectRequest(
            request_id=req.request_id,
            subject_id=req.subject_id,
            request_type=req.request_type,
            status=status,
            reason=req.reason,
            created_at=req.created_at,
            updated_at=now,
            completed_at=now,
            response_data={"errors": errors} if errors else {},
            affected_records=total_deleted,
            processor="auto",
            denial_reason=req.denial_reason,
            notes=req.notes,
        )
        self._requests[req.request_id] = updated
        return updated
