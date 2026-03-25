# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Evidence auto-collection from target data.

Provides ``collect_from_target()`` which gathers all available evidence
for a given target ID by querying in-memory data providers.  Callers
supply provider callbacks so there is no dependency on specific storage
backends or file I/O.

Usage::

    from tritium_lib.evidence import collect_from_target, EvidenceCollection

    collection = EvidenceCollection(title="Investigation: ble_aa:bb:cc")

    # Define data providers
    def get_signals(target_id):
        return [{"signal_type": "ble_advertisement", "mac": "aa:bb:cc", ...}]

    collected = collect_from_target(
        target_id="ble_aa:bb:cc",
        collection=collection,
        collector="analyst",
        signal_provider=get_signals,
    )
    print(f"Collected {len(collected)} evidence items")
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .models import (
    AssociationData,
    ClassificationData,
    Evidence,
    EvidenceType,
    SignalCaptureData,
    TrackLogData,
    TrackLogEntry,
    ZoneEventData,
)
from .collection import EvidenceCollection
from .integrity import hash_evidence


# Type alias for data provider callbacks
DataProvider = Callable[[str], list[dict[str, Any]]]


def _build_signal_evidence(
    target_id: str,
    record: dict[str, Any],
    collector: str,
) -> Evidence:
    """Build a signal capture evidence item from raw data.

    Args:
        target_id: Target this signal relates to.
        record: Raw signal data dict.
        collector: Who is collecting.

    Returns:
        Evidence item of type SIGNAL_CAPTURE.
    """
    return Evidence(
        evidence_type=EvidenceType.SIGNAL_CAPTURE,
        target_id=target_id,
        collected_by=collector,
        source_sensor=record.get("sensor_id", ""),
        data=record,
    )


def _build_track_evidence(
    target_id: str,
    records: list[dict[str, Any]],
    collector: str,
) -> Evidence:
    """Build a track log evidence item from position records.

    Args:
        target_id: Target this track relates to.
        records: List of position record dicts.
        collector: Who is collecting.

    Returns:
        Evidence item of type TRACK_LOG.
    """
    return Evidence(
        evidence_type=EvidenceType.TRACK_LOG,
        target_id=target_id,
        collected_by=collector,
        data={
            "target_id": target_id,
            "entries": records,
            "entry_count": len(records),
        },
    )


def _build_zone_evidence(
    target_id: str,
    record: dict[str, Any],
    collector: str,
) -> Evidence:
    """Build a zone event evidence item.

    Args:
        target_id: Target this zone event relates to.
        record: Zone event data dict.
        collector: Who is collecting.

    Returns:
        Evidence item of type ZONE_EVENT.
    """
    return Evidence(
        evidence_type=EvidenceType.ZONE_EVENT,
        target_id=target_id,
        collected_by=collector,
        data=record,
    )


def _build_association_evidence(
    target_id: str,
    record: dict[str, Any],
    collector: str,
) -> Evidence:
    """Build an association evidence item.

    Args:
        target_id: Primary target.
        record: Association data dict.
        collector: Who is collecting.

    Returns:
        Evidence item of type ASSOCIATION.
    """
    return Evidence(
        evidence_type=EvidenceType.ASSOCIATION,
        target_id=target_id,
        collected_by=collector,
        data=record,
    )


def _build_classification_evidence(
    target_id: str,
    record: dict[str, Any],
    collector: str,
) -> Evidence:
    """Build a classification evidence item.

    Args:
        target_id: Target that was classified.
        record: Classification result dict.
        collector: Who is collecting.

    Returns:
        Evidence item of type CLASSIFICATION.
    """
    return Evidence(
        evidence_type=EvidenceType.CLASSIFICATION,
        target_id=target_id,
        collected_by=collector,
        data=record,
    )


def collect_from_target(
    target_id: str,
    collection: EvidenceCollection,
    collector: str = "system",
    signal_provider: Optional[DataProvider] = None,
    track_provider: Optional[DataProvider] = None,
    zone_provider: Optional[DataProvider] = None,
    association_provider: Optional[DataProvider] = None,
    classification_provider: Optional[DataProvider] = None,
) -> list[Evidence]:
    """Auto-collect all available evidence for a target.

    Queries each provided data source and adds the results to the
    collection.  Providers that are None are skipped.

    Args:
        target_id: Target ID to collect evidence for.
        collection: Evidence collection to add items to.
        collector: Who is performing the collection.
        signal_provider: Returns list of signal capture dicts.
        track_provider: Returns list of position record dicts.
        zone_provider: Returns list of zone event dicts.
        association_provider: Returns list of association dicts.
        classification_provider: Returns list of classification dicts.

    Returns:
        List of Evidence items that were collected and added.
    """
    if target_id not in collection.target_ids:
        collection.target_ids.append(target_id)

    collected: list[Evidence] = []

    # Signal captures (one evidence item per signal)
    if signal_provider:
        signals = signal_provider(target_id)
        for sig in signals:
            ev = _build_signal_evidence(target_id, sig, collector)
            collection.add_evidence(ev, collector=collector)
            collected.append(ev)

    # Track log (one evidence item for entire track)
    if track_provider:
        positions = track_provider(target_id)
        if positions:
            ev = _build_track_evidence(target_id, positions, collector)
            collection.add_evidence(ev, collector=collector)
            collected.append(ev)

    # Zone events (one per event)
    if zone_provider:
        zone_events = zone_provider(target_id)
        for ze in zone_events:
            ev = _build_zone_evidence(target_id, ze, collector)
            collection.add_evidence(ev, collector=collector)
            collected.append(ev)

    # Associations (one per association)
    if association_provider:
        associations = association_provider(target_id)
        for assoc in associations:
            ev = _build_association_evidence(target_id, assoc, collector)
            collection.add_evidence(ev, collector=collector)
            collected.append(ev)

    # Classifications (one per classification result)
    if classification_provider:
        classifications = classification_provider(target_id)
        for cls_result in classifications:
            ev = _build_classification_evidence(target_id, cls_result, collector)
            collection.add_evidence(ev, collector=collector)
            collected.append(ev)

    return collected
