# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Data retention, anonymization, and privacy compliance.

Provides the tooling needed for GDPR and privacy-law compliance
when operating a surveillance/tracking system:

- **RetentionPolicy / RetentionManager** — define and enforce how long
  different categories of data are kept before automatic purge.
- **Anonymizer** — strip PII, pseudonymize identifiers, redact fields.
- **ConsentManager** — track data-processing consent per subject.
- **DataSubjectRequest / SubjectRequestManager** — handle GDPR-style
  right-of-access, right-to-erasure, and other subject requests.
- **PrivacyZone / PrivacyZoneManager** — geographic areas where tracking
  is suppressed or data is anonymized.

Default retention periods::

    realtime_sightings  :  7 days
    target_history      : 30 days
    dossiers            : 90 days
    incidents           :  1 year
    audit_trail         :  7 years

Usage
-----
::

    from tritium_lib.privacy import (
        RetentionManager,
        Anonymizer,
        ConsentManager,
        SubjectRequestManager,
        PrivacyZoneManager,
    )

    # Enforce retention policies
    retention = RetentionManager()
    retention.register_handler("realtime_sightings", my_purge_fn)
    results = retention.enforce()

    # Anonymize target data
    anon = Anonymizer(secret="deployment-secret")
    clean, meta = anon.anonymize_record(target_data)

    # Track consent
    consent = ConsentManager()
    consent.grant("target_123", "tracking", evidence="web_form")

    # Handle subject requests
    requests = SubjectRequestManager()
    req = requests.submit_access("target_123")
    requests.process(req.request_id)

    # Define privacy zones
    zones = PrivacyZoneManager()
    zones.add_zone("School", polygon, suppression="full")
    result = zones.check_point(40.7128, -74.0060)
"""

from __future__ import annotations

from .retention import (
    DataCategory,
    DEFAULT_RETENTION,
    RetentionPolicy,
    PurgeResult,
    PurgeHandler,
    RetentionManager,
)
from .anonymizer import (
    AnonymizationLevel,
    AnonymizationResult,
    Anonymizer,
    PII_FIELDS,
)
from .consent import (
    ProcessingPurpose,
    LegalBasis,
    ConsentStatus,
    ConsentRecord,
    ConsentManager,
)
from .subject_request import (
    RequestType,
    RequestStatus,
    DataSubjectRequest,
    SubjectRequestManager,
)
from .privacy_zone import (
    SuppressionLevel,
    PrivacyZone,
    PrivacyZoneManager,
    ZoneCheckResult,
)

__all__ = [
    # Retention
    "DataCategory",
    "DEFAULT_RETENTION",
    "RetentionPolicy",
    "PurgeResult",
    "PurgeHandler",
    "RetentionManager",
    # Anonymization
    "AnonymizationLevel",
    "AnonymizationResult",
    "Anonymizer",
    "PII_FIELDS",
    # Consent
    "ProcessingPurpose",
    "LegalBasis",
    "ConsentStatus",
    "ConsentRecord",
    "ConsentManager",
    # Subject requests
    "RequestType",
    "RequestStatus",
    "DataSubjectRequest",
    "SubjectRequestManager",
    # Privacy zones
    "SuppressionLevel",
    "PrivacyZone",
    "PrivacyZoneManager",
    "ZoneCheckResult",
]
