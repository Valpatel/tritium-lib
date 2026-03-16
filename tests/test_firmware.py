# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for firmware models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tritium_lib.models.firmware import FirmwareMeta, OTAJob, OTAStatus


def _utc(hour=12, minute=0, second=0):
    return datetime(2026, 3, 7, hour, minute, second, tzinfo=timezone.utc)


class TestOTAStatus:
    def test_all_values(self):
        expected = {"pending", "in_progress", "completed", "failed", "cancelled"}
        assert {s.value for s in OTAStatus} == expected

    def test_string_enum(self):
        assert OTAStatus.PENDING == "pending"
        assert OTAStatus.IN_PROGRESS == "in_progress"
        assert isinstance(OTAStatus.COMPLETED, str)


class TestFirmwareMeta:
    def test_create_minimal(self):
        fw = FirmwareMeta(id="fw-1", version="1.0.0")
        assert fw.id == "fw-1"
        assert fw.version == "1.0.0"
        assert fw.board == "any"
        assert fw.family == "esp32"
        assert fw.size == 0
        assert fw.sha256 == ""
        assert fw.signed is False
        assert fw.encrypted is False
        assert fw.notes == ""

    def test_create_full(self):
        fw = FirmwareMeta(
            id="fw-2",
            version="2.1.0",
            board="touch-lcd-349",
            family="esp32",
            size=1048576,
            sha256="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            signed=True,
            encrypted=False,
            build_timestamp=_utc(8),
            uploaded_at=_utc(10),
            notes="Production release",
        )
        assert fw.board == "touch-lcd-349"
        assert fw.size == 1048576
        assert fw.signed is True
        assert fw.notes == "Production release"

    def test_serialization(self):
        fw = FirmwareMeta(
            id="fw-3",
            version="1.5.0",
            board="touch-amoled-241b",
            size=524288,
            sha256="deadbeef",
            signed=True,
        )
        d = fw.model_dump()
        assert d["version"] == "1.5.0"
        assert d["signed"] is True
        assert d["encrypted"] is False

    def test_json_roundtrip(self):
        fw = FirmwareMeta(
            id="fw-4",
            version="3.0.0",
            board="touch-lcd-35bc",
            size=2097152,
            sha256="cafebabe",
            build_timestamp=_utc(),
        )
        json_str = fw.model_dump_json()
        fw2 = FirmwareMeta.model_validate_json(json_str)
        assert fw2.id == fw.id
        assert fw2.version == fw.version
        assert fw2.size == fw.size
        assert fw2.sha256 == fw.sha256

    def test_none_optional_fields(self):
        fw = FirmwareMeta(id="fw-5", version="0.1.0")
        assert fw.build_timestamp is None
        assert fw.uploaded_at is None

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            FirmwareMeta(id="fw-6")  # missing version
        with pytest.raises(ValidationError):
            FirmwareMeta(version="1.0.0")  # missing id


class TestOTAJob:
    def test_create_minimal(self):
        job = OTAJob(id="ota-1", firmware_url="https://example.com/fw.bin")
        assert job.id == "ota-1"
        assert job.firmware_url == "https://example.com/fw.bin"
        assert job.target_devices == []
        assert job.status == OTAStatus.PENDING
        assert job.firmware_version == ""
        assert job.firmware_sha256 == ""
        assert job.error is None

    def test_create_full(self):
        job = OTAJob(
            id="ota-2",
            firmware_url="https://example.com/fw-v2.bin",
            target_devices=["dev-a", "dev-b", "dev-c"],
            status=OTAStatus.IN_PROGRESS,
            created_at=_utc(10),
            completed_at=None,
            firmware_version="2.0.0",
            firmware_sha256="abc123def456",
            error=None,
        )
        assert len(job.target_devices) == 3
        assert job.status == OTAStatus.IN_PROGRESS

    def test_all_statuses(self):
        for st in OTAStatus:
            job = OTAJob(
                id="t", firmware_url="http://x.com/f.bin", status=st
            )
            assert job.status == st

    def test_serialization(self):
        job = OTAJob(
            id="ota-3",
            firmware_url="https://example.com/fw.bin",
            target_devices=["dev-1"],
            status=OTAStatus.COMPLETED,
            firmware_version="1.0.0",
        )
        d = job.model_dump()
        assert d["status"] == "completed"
        assert d["target_devices"] == ["dev-1"]

    def test_json_roundtrip(self):
        job = OTAJob(
            id="ota-4",
            firmware_url="https://example.com/fw.bin",
            target_devices=["dev-x", "dev-y"],
            status=OTAStatus.FAILED,
            created_at=_utc(),
            error="Network timeout",
        )
        json_str = job.model_dump_json()
        job2 = OTAJob.model_validate_json(json_str)
        assert job2.id == job.id
        assert job2.status == OTAStatus.FAILED
        assert job2.error == "Network timeout"
        assert job2.target_devices == job.target_devices

    def test_empty_targets(self):
        job = OTAJob(id="ota-5", firmware_url="http://x.com/f.bin")
        assert job.target_devices == []

    def test_completed_job(self):
        job = OTAJob(
            id="ota-6",
            firmware_url="https://example.com/fw.bin",
            target_devices=["dev-a"],
            status=OTAStatus.COMPLETED,
            created_at=_utc(8),
            completed_at=_utc(8, 15),
            firmware_version="1.2.3",
            firmware_sha256="deadbeef",
        )
        assert job.completed_at is not None
        assert job.error is None

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            OTAJob(id="ota-7")  # missing firmware_url

    def test_from_dict(self):
        data = {
            "id": "ota-8",
            "firmware_url": "http://x.com/f.bin",
            "status": "cancelled",
        }
        job = OTAJob.model_validate(data)
        assert job.status == OTAStatus.CANCELLED
