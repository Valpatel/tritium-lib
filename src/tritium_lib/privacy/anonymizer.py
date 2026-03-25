# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Data anonymization utilities for privacy compliance.

Provides :class:`Anonymizer` which strips PII (personally identifiable
information) from target data, hashes identifiers for pseudonymization,
and redacts sensitive fields.

Supports two modes:
    - **Pseudonymization**: replace identifiers with deterministic hashes
      (reversible with the secret key).
    - **Full anonymization**: irreversible removal / generalization of data.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("privacy.anonymizer")


# ---------------------------------------------------------------------------
# Anonymization level
# ---------------------------------------------------------------------------

class AnonymizationLevel(str, Enum):
    """How aggressively to strip identifying information."""
    NONE = "none"
    PSEUDONYMIZE = "pseudonymize"  # hash identifiers, keep structure
    ANONYMIZE = "anonymize"        # remove all PII, generalize data
    REDACT = "redact"              # replace all sensitive fields with [REDACTED]


# ---------------------------------------------------------------------------
# Field classification
# ---------------------------------------------------------------------------

# Fields considered PII and subject to anonymization
PII_FIELDS: set[str] = {
    "mac_address", "mac", "bssid", "ssid",
    "device_name", "name", "hostname",
    "ip_address", "ip", "source_ip",
    "phone", "email", "imei", "imsi",
    "plate_number", "license_plate",
    "bluetooth_name", "ble_name",
    "face_encoding", "face_id",
    "owner", "operator",
}

# Fields that contain location data
LOCATION_FIELDS: set[str] = {
    "lat", "lng", "latitude", "longitude",
    "position", "location", "gps",
    "x", "y",
}

# Regex patterns for MAC addresses
MAC_RE = re.compile(r"([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}")

# Regex for IPv4
IPV4_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


# ---------------------------------------------------------------------------
# AnonymizationResult
# ---------------------------------------------------------------------------

@dataclass
class AnonymizationResult:
    """Metadata about one anonymization operation."""
    original_field_count: int = 0
    anonymized_field_count: int = 0
    level: str = "none"
    timestamp: float = field(default_factory=time.time)

    @property
    def fields_affected(self) -> int:
        return self.anonymized_field_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_field_count": self.original_field_count,
            "anonymized_field_count": self.anonymized_field_count,
            "level": self.level,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Anonymizer
# ---------------------------------------------------------------------------

class Anonymizer:
    """Strip PII from target data, hash identifiers, redact sensitive fields.

    Parameters
    ----------
    secret :
        HMAC secret used for pseudonymization hashing.  Different secrets
        produce different hashes, allowing per-deployment isolation.
    default_level :
        Default anonymization level if not specified per-call.
    hash_algorithm :
        Hash function for pseudonymization (default: sha256).
    location_precision :
        Decimal places to keep for location anonymization.
        3 = ~111m accuracy, 2 = ~1.1km, 1 = ~11km.
    extra_pii_fields :
        Additional field names to treat as PII.
    """

    def __init__(
        self,
        secret: str = "tritium-default-secret",
        default_level: AnonymizationLevel = AnonymizationLevel.PSEUDONYMIZE,
        hash_algorithm: str = "sha256",
        location_precision: int = 3,
        extra_pii_fields: Optional[set[str]] = None,
    ) -> None:
        self._secret = secret.encode("utf-8")
        self._default_level = default_level
        self._hash_algorithm = hash_algorithm
        self._location_precision = location_precision
        self._pii_fields = PII_FIELDS | (extra_pii_fields or set())
        self._stats_count = 0
        self._stats_fields = 0

    # -- public API ---------------------------------------------------------

    def anonymize_record(
        self,
        record: dict[str, Any],
        level: Optional[AnonymizationLevel] = None,
    ) -> tuple[dict[str, Any], AnonymizationResult]:
        """Anonymize a single data record (dict).

        Returns (anonymized_copy, result_metadata).  The original is
        never modified.
        """
        level = level or self._default_level
        if level == AnonymizationLevel.NONE:
            return dict(record), AnonymizationResult(
                original_field_count=len(record),
                level=level.value,
            )

        out: dict[str, Any] = {}
        anonymized_count = 0

        for key, value in record.items():
            lower_key = key.lower()

            if lower_key in self._pii_fields:
                out[key] = self._anonymize_value(key, value, level)
                anonymized_count += 1
            elif lower_key in LOCATION_FIELDS and level == AnonymizationLevel.ANONYMIZE:
                out[key] = self._generalize_location(value)
                anonymized_count += 1
            elif isinstance(value, str):
                # Scrub embedded MACs / IPs from free-text fields
                scrubbed = self._scrub_text(value, level)
                if scrubbed != value:
                    anonymized_count += 1
                out[key] = scrubbed
            else:
                out[key] = value

        result = AnonymizationResult(
            original_field_count=len(record),
            anonymized_field_count=anonymized_count,
            level=level.value,
        )
        self._stats_count += 1
        self._stats_fields += anonymized_count
        return out, result

    def anonymize_identifier(self, identifier: str) -> str:
        """Pseudonymize a single identifier string using HMAC."""
        return self._hash_value(identifier)

    def anonymize_mac(self, mac: str) -> str:
        """Hash a MAC address for pseudonymization."""
        normalized = mac.upper().replace("-", ":")
        return f"anon_{self._hash_value(normalized)[:12]}"

    def hash_target_id(self, target_id: str) -> str:
        """Create a pseudonymized target ID preserving the prefix.

        ``ble_AA:BB:CC:DD:EE:FF`` becomes ``ble_anon_<hash>``
        """
        if "_" in target_id:
            prefix, rest = target_id.split("_", 1)
            return f"{prefix}_anon_{self._hash_value(rest)[:16]}"
        return f"anon_{self._hash_value(target_id)[:16]}"

    def is_pii_field(self, field_name: str) -> bool:
        """Check if a field name is classified as PII."""
        return field_name.lower() in self._pii_fields

    def add_pii_field(self, field_name: str) -> None:
        """Register an additional field as PII."""
        self._pii_fields.add(field_name.lower())

    @property
    def stats(self) -> dict[str, int]:
        """Return anonymization statistics."""
        return {
            "records_processed": self._stats_count,
            "fields_anonymized": self._stats_fields,
        }

    def reset_stats(self) -> None:
        """Reset running statistics."""
        self._stats_count = 0
        self._stats_fields = 0

    # -- internals ----------------------------------------------------------

    def _hash_value(self, value: str) -> str:
        """HMAC hash of a value using the secret."""
        h = hmac.new(self._secret, value.encode("utf-8"), self._hash_algorithm)
        return h.hexdigest()

    def _anonymize_value(
        self, key: str, value: Any, level: AnonymizationLevel
    ) -> Any:
        """Anonymize a single PII field value."""
        if level == AnonymizationLevel.REDACT:
            return "[REDACTED]"

        if value is None:
            return None

        if isinstance(value, str):
            if level == AnonymizationLevel.PSEUDONYMIZE:
                return self._hash_value(value)[:16]
            # ANONYMIZE
            return "[REMOVED]"

        # Non-string PII (rare) — just redact
        if level == AnonymizationLevel.ANONYMIZE:
            return None
        return "[REDACTED]"

    def _generalize_location(self, value: Any) -> Any:
        """Reduce location precision for anonymization."""
        if isinstance(value, (int, float)):
            return round(float(value), self._location_precision)
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                if k.lower() in LOCATION_FIELDS and isinstance(v, (int, float)):
                    out[k] = round(float(v), self._location_precision)
                else:
                    out[k] = v
            return out
        return value

    def _scrub_text(self, text: str, level: AnonymizationLevel) -> str:
        """Remove embedded MACs and IPs from free-text strings."""
        if level == AnonymizationLevel.REDACT:
            text = MAC_RE.sub("[MAC_REDACTED]", text)
            text = IPV4_RE.sub("[IP_REDACTED]", text)
            return text
        if level == AnonymizationLevel.ANONYMIZE:
            text = MAC_RE.sub("[MAC_REMOVED]", text)
            text = IPV4_RE.sub("[IP_REMOVED]", text)
            return text
        # Pseudonymize — replace MACs with hashed versions
        def _replace_mac(m: re.Match) -> str:
            return self.anonymize_mac(m.group(0))

        text = MAC_RE.sub(_replace_mac, text)
        return text
