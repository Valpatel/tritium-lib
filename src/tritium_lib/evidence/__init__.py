# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.evidence — Evidence collection and chain-of-custody tracking.

Collect, preserve, verify, and export evidence for investigations with
proper chain-of-custody records and SHA-256 integrity hashing.

Core classes:
  - Evidence             — individual piece of evidence (signal, track, zone event)
  - EvidenceChain        — chain of custody record (who accessed when)
  - EvidenceCollection   — grouped evidence for an investigation
  - EvidenceExporter     — export evidence package with manifest
  - Integrity functions  — SHA-256 hash computation and verification

Quick start::

    from tritium_lib.evidence import (
        Evidence, EvidenceType, EvidenceCollection, EvidenceExporter,
        hash_evidence, verify_integrity, collect_from_target,
    )

    # Create a collection for an investigation
    collection = EvidenceCollection(
        title="Hostile device investigation",
        created_by="analyst",
        target_ids=["ble_aa:bb:cc:dd"],
    )

    # Add evidence manually
    ev = Evidence(
        evidence_type=EvidenceType.SIGNAL_CAPTURE,
        target_id="ble_aa:bb:cc:dd",
        collected_by="analyst",
        source_sensor="edge-01",
        data={"signal_type": "ble_advertisement", "rssi": -65},
    )
    collection.add_evidence(ev, collector="analyst")

    # Verify integrity of all evidence
    results = collection.verify_all()

    # Export the collection
    exporter = EvidenceExporter()
    entries = exporter.export_collection(collection, actor="analyst")

    # Auto-collect evidence for a target
    collected = collect_from_target(
        target_id="ble_aa:bb:cc:dd",
        collection=collection,
        collector="analyst",
        signal_provider=lambda tid: [{"signal_type": "wifi_probe", "rssi": -70}],
    )
"""

from .models import (
    AssociationData,
    ClassificationData,
    Evidence,
    EvidenceStatus,
    EvidenceType,
    SignalCaptureData,
    TrackLogData,
    TrackLogEntry,
    ZoneEventData,
)
from .chain import (
    CustodyAction,
    CustodyEvent,
    EvidenceChain,
)
from .collection import (
    EvidenceCollection,
    InvestigationStatus,
)
from .integrity import (
    compute_sha256,
    hash_bytes,
    hash_evidence,
    verify_hash,
    verify_integrity,
)
from .exporter import (
    EvidenceExporter,
    ExportEntry,
)
from .collector import (
    collect_from_target,
)

__all__ = [
    # Models
    "Evidence",
    "EvidenceType",
    "EvidenceStatus",
    "SignalCaptureData",
    "TrackLogData",
    "TrackLogEntry",
    "ZoneEventData",
    "AssociationData",
    "ClassificationData",
    # Chain of custody
    "CustodyAction",
    "CustodyEvent",
    "EvidenceChain",
    # Collection
    "EvidenceCollection",
    "InvestigationStatus",
    # Integrity
    "compute_sha256",
    "hash_bytes",
    "hash_evidence",
    "verify_hash",
    "verify_integrity",
    # Export
    "EvidenceExporter",
    "ExportEntry",
    # Collector
    "collect_from_target",
]
