# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.deployment — deployment automation module."""

from __future__ import annotations

import json
import os
import signal
import tempfile
import time

import pytest

from tritium_lib.deployment import (
    BackupManager,
    BackupManifest,
    ComponentStatus,
    DeploymentComponent,
    DeploymentConfig,
    HealthCheck,
    LogCollector,
    LogEntry,
    RequirementsResult,
    SystemRequirements,
)
from tritium_lib.deployment.config import (
    ComponentType,
    DeploymentEnvironment,
    _DEFAULT_PORTS,
)
from tritium_lib.deployment.status import StatusLevel
from tritium_lib.deployment.health import HealthReport
from tritium_lib.deployment.logs import LogLevel


# ---------------------------------------------------------------------------
# DeploymentComponent
# ---------------------------------------------------------------------------

class TestDeploymentComponent:
    """Tests for DeploymentComponent data model."""

    def test_default_values(self):
        comp = DeploymentComponent(name="sc")
        assert comp.name == "sc"
        assert comp.component_type == ComponentType.SC
        assert comp.version == "0.1.0"
        assert comp.port == 0
        assert comp.config_overrides == {}
        assert comp.enabled is True

    def test_custom_values(self):
        comp = DeploymentComponent(
            name="mqtt",
            component_type=ComponentType.MQTT,
            version="2.0.0",
            port=1883,
            config_overrides={"max_connections": 100},
            enabled=False,
        )
        assert comp.name == "mqtt"
        assert comp.component_type == ComponentType.MQTT
        assert comp.version == "2.0.0"
        assert comp.port == 1883
        assert comp.config_overrides == {"max_connections": 100}
        assert comp.enabled is False

    def test_to_dict(self):
        comp = DeploymentComponent(name="edge", component_type=ComponentType.EDGE)
        d = comp.to_dict()
        assert d["name"] == "edge"
        assert d["component_type"] == "edge"
        assert d["enabled"] is True

    def test_from_dict(self):
        data = {
            "name": "database",
            "component_type": "database",
            "version": "3.0.0",
            "port": 5432,
            "config_overrides": {"pool_size": 10},
            "enabled": True,
        }
        comp = DeploymentComponent.from_dict(data)
        assert comp.name == "database"
        assert comp.component_type == ComponentType.DATABASE
        assert comp.port == 5432

    def test_roundtrip_serialization(self):
        original = DeploymentComponent(
            name="addons",
            component_type=ComponentType.ADDONS,
            version="1.2.3",
            port=9000,
            config_overrides={"debug": True},
        )
        restored = DeploymentComponent.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.component_type == original.component_type
        assert restored.version == original.version
        assert restored.port == original.port
        assert restored.config_overrides == original.config_overrides


# ---------------------------------------------------------------------------
# DeploymentConfig
# ---------------------------------------------------------------------------

class TestDeploymentConfig:
    """Tests for DeploymentConfig data model."""

    def test_default_config(self):
        cfg = DeploymentConfig(name="test")
        assert cfg.name == "test"
        assert cfg.host == "localhost"
        assert cfg.environment == DeploymentEnvironment.DEVELOPMENT
        assert cfg.mqtt_port == 1883

    def test_auto_component_specs(self):
        cfg = DeploymentConfig(
            name="test",
            components=["sc", "mqtt"],
        )
        assert len(cfg.component_specs) == 2
        assert cfg.component_specs[0].name == "sc"
        assert cfg.component_specs[0].port == _DEFAULT_PORTS["sc"]
        assert cfg.component_specs[1].name == "mqtt"
        assert cfg.component_specs[1].port == _DEFAULT_PORTS["mqtt"]

    def test_enabled_components(self):
        cfg = DeploymentConfig(name="test")
        cfg.component_specs = [
            DeploymentComponent(name="sc", enabled=True),
            DeploymentComponent(name="edge", enabled=False),
            DeploymentComponent(name="mqtt", enabled=True),
        ]
        enabled = cfg.enabled_components
        assert len(enabled) == 2
        assert enabled[0].name == "sc"
        assert enabled[1].name == "mqtt"

    def test_component_names(self):
        cfg = DeploymentConfig(name="test", components=["sc", "mqtt", "edge"])
        assert cfg.component_names == ["sc", "mqtt", "edge"]

    def test_get_component(self):
        cfg = DeploymentConfig(name="test", components=["sc", "mqtt"])
        assert cfg.get_component("sc") is not None
        assert cfg.get_component("sc").name == "sc"
        assert cfg.get_component("nonexistent") is None

    def test_validate_valid(self):
        cfg = DeploymentConfig(name="test", components=["sc"])
        errors = cfg.validate()
        assert errors == []

    def test_validate_missing_name(self):
        cfg = DeploymentConfig(name="", components=["sc"])
        errors = cfg.validate()
        assert any("name" in e.lower() for e in errors)

    def test_validate_no_components(self):
        cfg = DeploymentConfig(name="test")
        errors = cfg.validate()
        assert any("component" in e.lower() for e in errors)

    def test_validate_invalid_port(self):
        cfg = DeploymentConfig(name="test", components=["sc"], mqtt_port=0)
        errors = cfg.validate()
        assert any("port" in e.lower() for e in errors)

    def test_validate_duplicate_components(self):
        cfg = DeploymentConfig(name="test")
        cfg.component_specs = [
            DeploymentComponent(name="sc"),
            DeploymentComponent(name="sc"),
        ]
        errors = cfg.validate()
        assert any("duplicate" in e.lower() for e in errors)

    def test_to_dict_excludes_credentials(self):
        cfg = DeploymentConfig(
            name="test",
            credentials={"api_key": "secret123"},
            components=["sc"],
        )
        d = cfg.to_dict()
        assert "credentials" not in d

    def test_roundtrip_serialization(self):
        original = DeploymentConfig(
            name="field-1",
            host="192.168.1.100",
            environment=DeploymentEnvironment.FIELD,
            components=["sc", "mqtt"],
            mqtt_broker="192.168.1.1",
            mqtt_port=1883,
            tags={"region": "north"},
        )
        restored = DeploymentConfig.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.host == original.host
        assert restored.environment == original.environment
        assert restored.mqtt_broker == original.mqtt_broker
        assert restored.tags == original.tags


# ---------------------------------------------------------------------------
# ComponentStatus
# ---------------------------------------------------------------------------

class TestComponentStatus:
    """Tests for ComponentStatus model."""

    def test_default_status(self):
        cs = ComponentStatus(name="sc")
        assert cs.name == "sc"
        assert cs.status == StatusLevel.UNKNOWN
        assert cs.pid == 0
        assert cs.is_running is False
        assert cs.is_healthy is False

    def test_running_status(self):
        cs = ComponentStatus(name="sc", status=StatusLevel.RUNNING, pid=1234)
        assert cs.is_running is True
        assert cs.is_healthy is True
        assert cs.needs_attention is False

    def test_error_status(self):
        cs = ComponentStatus(
            name="mqtt",
            status=StatusLevel.ERROR,
            error_message="Connection refused",
        )
        assert cs.is_running is False
        assert cs.needs_attention is True

    def test_degraded_needs_attention(self):
        cs = ComponentStatus(name="sc", status=StatusLevel.DEGRADED)
        assert cs.needs_attention is True
        assert cs.is_healthy is False

    def test_starting_is_healthy(self):
        cs = ComponentStatus(name="sc", status=StatusLevel.STARTING)
        assert cs.is_healthy is True
        assert cs.is_running is False

    def test_to_dict(self):
        cs = ComponentStatus(
            name="sc",
            status=StatusLevel.RUNNING,
            pid=42,
            version="0.1.0",
        )
        d = cs.to_dict()
        assert d["name"] == "sc"
        assert d["status"] == "running"
        assert d["pid"] == 42

    def test_from_dict(self):
        data = {"name": "mqtt", "status": "error", "error_message": "fail"}
        cs = ComponentStatus.from_dict(data)
        assert cs.name == "mqtt"
        assert cs.status == StatusLevel.ERROR
        assert cs.error_message == "fail"

    def test_str_representation(self):
        cs = ComponentStatus(
            name="sc",
            status=StatusLevel.RUNNING,
            pid=100,
            version="0.1.0",
        )
        s = str(cs)
        assert "sc" in s
        assert "running" in s
        assert "pid=100" in s


# ---------------------------------------------------------------------------
# HealthCheck
# ---------------------------------------------------------------------------

class TestHealthCheck:
    """Tests for HealthCheck — local health verification."""

    def test_check_unknown_component(self):
        cfg = DeploymentConfig(name="test", components=["sc"])
        hc = HealthCheck(cfg)
        status = hc.check_component("nonexistent")
        assert status.status == StatusLevel.UNKNOWN
        assert "not in config" in status.error_message

    def test_check_component_no_pid_file(self):
        with tempfile.TemporaryDirectory() as pid_dir:
            cfg = DeploymentConfig(name="test", components=["sc"])
            hc = HealthCheck(cfg, pid_dir=pid_dir)
            status = hc.check_component("sc")
            assert status.status == StatusLevel.STOPPED
            assert "No PID" in status.error_message

    def test_check_component_stale_pid(self):
        """PID file exists but process is not running."""
        with tempfile.TemporaryDirectory() as pid_dir:
            pid_file = os.path.join(pid_dir, "sc.pid")
            with open(pid_file, "w") as f:
                f.write("999999999")  # PID that almost certainly doesn't exist

            cfg = DeploymentConfig(name="test", components=["sc"])
            hc = HealthCheck(cfg, pid_dir=pid_dir)
            status = hc.check_component("sc")
            assert status.status == StatusLevel.ERROR
            assert "not running" in status.error_message

    def test_check_component_valid_pid(self):
        """PID file exists and points to our own process."""
        with tempfile.TemporaryDirectory() as pid_dir:
            pid_file = os.path.join(pid_dir, "sc.pid")
            with open(pid_file, "w") as f:
                f.write(str(os.getpid()))  # Our own PID — guaranteed to exist

            cfg = DeploymentConfig(name="test", components=["sc"])
            hc = HealthCheck(cfg, pid_dir=pid_dir)
            status = hc.check_component("sc")
            assert status.status == StatusLevel.RUNNING
            assert status.pid == os.getpid()

    def test_check_all_report(self):
        with tempfile.TemporaryDirectory() as pid_dir:
            cfg = DeploymentConfig(name="test", components=["sc", "mqtt"])
            hc = HealthCheck(cfg, pid_dir=pid_dir)
            report = hc.check_all()
            assert isinstance(report, HealthReport)
            assert report.deployment_name == "test"
            assert report.component_count == 2
            assert report.overall == "unhealthy"  # No PID files exist

    def test_check_all_healthy(self):
        """All components have valid PIDs."""
        with tempfile.TemporaryDirectory() as pid_dir:
            for name in ["sc", "mqtt"]:
                with open(os.path.join(pid_dir, f"{name}.pid"), "w") as f:
                    f.write(str(os.getpid()))

            cfg = DeploymentConfig(name="test", components=["sc", "mqtt"])
            hc = HealthCheck(cfg, pid_dir=pid_dir)
            report = hc.check_all()
            assert report.overall == "healthy"
            assert report.running_count == 2

    def test_check_all_degraded(self):
        """One component running, one not."""
        with tempfile.TemporaryDirectory() as pid_dir:
            with open(os.path.join(pid_dir, "sc.pid"), "w") as f:
                f.write(str(os.getpid()))
            # mqtt has no PID file

            cfg = DeploymentConfig(name="test", components=["sc", "mqtt"])
            hc = HealthCheck(cfg, pid_dir=pid_dir)
            report = hc.check_all()
            assert report.overall == "degraded"
            assert report.running_count == 1

    def test_health_report_to_dict(self):
        report = HealthReport(deployment_name="test")
        report.component_statuses["sc"] = ComponentStatus(
            name="sc", status=StatusLevel.RUNNING
        )
        report.overall = "healthy"
        d = report.to_dict()
        assert d["deployment_name"] == "test"
        assert d["overall"] == "healthy"
        assert d["component_count"] == 1
        assert d["running_count"] == 1

    def test_check_data_dir(self):
        with tempfile.TemporaryDirectory() as data_dir:
            cfg = DeploymentConfig(name="test", data_dir=data_dir)
            hc = HealthCheck(cfg)
            assert hc.check_data_dir() is True

    def test_check_data_dir_missing(self):
        cfg = DeploymentConfig(name="test", data_dir="/nonexistent/path")
        hc = HealthCheck(cfg)
        assert hc.check_data_dir() is False


# ---------------------------------------------------------------------------
# SystemRequirements
# ---------------------------------------------------------------------------

class TestSystemRequirements:
    """Tests for SystemRequirements — local system checks."""

    def test_python_version_passes(self):
        reqs = SystemRequirements(min_python=(3, 10))
        check = reqs.check_python_version()
        assert check.passed is True
        assert check.name == "python_version"

    def test_python_version_too_high(self):
        reqs = SystemRequirements(min_python=(99, 0))
        check = reqs.check_python_version()
        assert check.passed is False

    def test_disk_space_passes(self):
        reqs = SystemRequirements(min_disk_gb=0.001)
        check = reqs.check_disk_space()
        assert check.passed is True
        assert "GB" in check.actual

    def test_disk_space_ridiculous_requirement(self):
        reqs = SystemRequirements(min_disk_gb=999999999)
        check = reqs.check_disk_space()
        assert check.passed is False

    def test_memory_check(self):
        reqs = SystemRequirements(min_memory_mb=1)
        check = reqs.check_memory()
        # Should pass on Linux (where /proc/meminfo exists)
        assert check.passed is True

    def test_platform_check(self):
        reqs = SystemRequirements()
        check = reqs.check_platform()
        assert check.passed is True
        assert check.name == "platform"
        assert check.actual != ""

    def test_directory_check_exists(self):
        with tempfile.TemporaryDirectory() as d:
            reqs = SystemRequirements()
            check = reqs.check_directory(d)
            assert check.passed is True

    def test_directory_check_missing(self):
        reqs = SystemRequirements()
        check = reqs.check_directory("/nonexistent/dir/path")
        assert check.passed is False
        assert "missing" in check.actual

    def test_check_local_full(self):
        reqs = SystemRequirements(
            min_python=(3, 10),
            min_disk_gb=0.001,
            min_memory_mb=1,
        )
        result = reqs.check_local()
        assert isinstance(result, RequirementsResult)
        assert result.meets_minimum is True
        assert result.passed_count >= 3  # python, disk, memory, platform

    def test_check_local_with_required_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            reqs = SystemRequirements(required_dirs=[d, "/nonexistent/path"])
            result = reqs.check_local()
            assert result.failed_count >= 1
            assert result.meets_minimum is False

    def test_requirements_result_to_dict(self):
        reqs = SystemRequirements(min_python=(3, 10))
        result = reqs.check_local()
        d = result.to_dict()
        assert "meets_minimum" in d
        assert "checks" in d
        assert isinstance(d["checks"], list)

    def test_requirements_result_failures(self):
        reqs = SystemRequirements(
            min_python=(99, 0),
            required_dirs=["/no/such/path"],
        )
        result = reqs.check_local()
        failures = result.failures
        assert len(failures) >= 1
        assert all(not f.passed for f in failures)


# ---------------------------------------------------------------------------
# BackupManager
# ---------------------------------------------------------------------------

class TestBackupManager:
    """Tests for BackupManager — local backup creation and restore."""

    def _make_source(self, tmp_dir: str) -> str:
        """Create a source directory with test files."""
        src = os.path.join(tmp_dir, "source")
        os.makedirs(src)
        for i in range(3):
            with open(os.path.join(src, f"file{i}.txt"), "w") as f:
                f.write(f"data-{i}\n" * 10)
        sub = os.path.join(src, "subdir")
        os.makedirs(sub)
        with open(os.path.join(sub, "nested.txt"), "w") as f:
            f.write("nested data")
        return src

    def test_create_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self._make_source(tmp)
            backup_dir = os.path.join(tmp, "backups")
            bm = BackupManager(backup_dir=backup_dir)
            manifest = bm.create_backup(src, name="test-backup")
            assert manifest.file_count == 4
            assert manifest.total_bytes > 0
            assert manifest.name == "test-backup"
            assert os.path.isfile(
                os.path.join(manifest.backup_path, "manifest.json")
            )

    def test_create_backup_missing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            bm = BackupManager(backup_dir=os.path.join(tmp, "backups"))
            with pytest.raises(FileNotFoundError):
                bm.create_backup("/nonexistent/source")

    def test_restore_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self._make_source(tmp)
            backup_dir = os.path.join(tmp, "backups")
            bm = BackupManager(backup_dir=backup_dir)
            manifest = bm.create_backup(src)

            target = os.path.join(tmp, "restored")
            restored_manifest = bm.restore_backup(manifest.backup_id, target)
            assert restored_manifest.file_count == manifest.file_count
            assert os.path.isfile(os.path.join(target, "file0.txt"))
            assert os.path.isfile(os.path.join(target, "subdir", "nested.txt"))

    def test_restore_nonexistent_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            bm = BackupManager(backup_dir=os.path.join(tmp, "backups"))
            with pytest.raises(FileNotFoundError):
                bm.restore_backup("fake-id", os.path.join(tmp, "target"))

    def test_list_backups(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self._make_source(tmp)
            backup_dir = os.path.join(tmp, "backups")
            bm = BackupManager(backup_dir=backup_dir)
            bm.create_backup(src, name="first")
            bm.create_backup(src, name="second")
            backups = bm.list_backups()
            assert len(backups) == 2
            # Newest first
            assert backups[0].created_at >= backups[1].created_at

    def test_delete_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self._make_source(tmp)
            backup_dir = os.path.join(tmp, "backups")
            bm = BackupManager(backup_dir=backup_dir)
            manifest = bm.create_backup(src)
            assert bm.delete_backup(manifest.backup_id) is True
            assert bm.list_backups() == []
            assert bm.delete_backup("nonexistent") is False

    def test_prune_old_backups(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self._make_source(tmp)
            backup_dir = os.path.join(tmp, "backups")
            bm = BackupManager(backup_dir=backup_dir, max_backups=2)
            bm.create_backup(src, name="a")
            bm.create_backup(src, name="b")
            bm.create_backup(src, name="c")
            backups = bm.list_backups()
            assert len(backups) <= 2

    def test_manifest_checksum(self):
        m = BackupManifest(name="test", file_count=5, total_bytes=1024)
        cs = m.compute_checksum()
        assert cs != ""
        assert m.verify_checksum() is True

    def test_manifest_checksum_tampered(self):
        m = BackupManifest(name="test", file_count=5)
        m.compute_checksum()
        m.file_count = 999  # Tamper
        assert m.verify_checksum() is False

    def test_manifest_size_mb(self):
        m = BackupManifest(total_bytes=1024 * 1024 * 10)
        assert abs(m.size_mb - 10.0) < 0.01

    def test_manifest_roundtrip(self):
        original = BackupManifest(
            name="test",
            source_dir="/data",
            file_count=42,
            total_bytes=99999,
            tags={"env": "prod"},
            files=["a.txt", "b.txt"],
        )
        original.compute_checksum()
        restored = BackupManifest.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.file_count == original.file_count
        assert restored.tags == original.tags
        assert restored.files == original.files
        assert restored.checksum == original.checksum

    def test_list_backups_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            bm = BackupManager(backup_dir=tmp)
            assert bm.list_backups() == []

    def test_list_backups_missing_dir(self):
        bm = BackupManager(backup_dir="/nonexistent/backup/dir")
        assert bm.list_backups() == []


# ---------------------------------------------------------------------------
# LogCollector
# ---------------------------------------------------------------------------

class TestLogCollector:
    """Tests for LogCollector — local log collection and parsing."""

    def _make_log_file(self, directory: str, name: str, lines: list[str]) -> str:
        path = os.path.join(directory, name)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def test_find_log_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_log_file(tmp, "sc.log", ["test"])
            self._make_log_file(tmp, "mqtt.log", ["test"])
            self._make_log_file(tmp, "notes.md", ["test"])  # Should be excluded
            lc = LogCollector(log_dirs=[tmp])
            files = lc.find_log_files()
            assert len(files) == 2
            assert all(f.endswith(".log") for f in files)

    def test_parse_line_valid(self):
        lc = LogCollector()
        line = "2026-03-25 10:30:45 - tritium.sc - INFO - Server started"
        entry = lc.parse_line(line)
        assert entry is not None
        assert entry.level == LogLevel.INFO
        assert entry.component == "tritium.sc"
        assert entry.message == "Server started"

    def test_parse_line_error(self):
        lc = LogCollector()
        line = "2026-03-25 10:30:45 - mqtt - ERROR - Connection lost"
        entry = lc.parse_line(line)
        assert entry is not None
        assert entry.level == LogLevel.ERROR
        assert "Connection lost" in entry.message

    def test_parse_line_invalid(self):
        lc = LogCollector()
        entry = lc.parse_line("this is not a log line")
        assert entry is None

    def test_collect_from_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            lines = [
                f"{now} - sc - INFO - Boot complete",
                f"{now} - sc - ERROR - Disk full",
                "garbage line",
                f"{now} - mqtt - WARNING - Slow broker",
            ]
            self._make_log_file(tmp, "app.log", lines)
            lc = LogCollector(log_dirs=[tmp])
            entries = lc.collect(since_hours=1)
            assert len(entries) == 3  # garbage line skipped

    def test_collect_with_level_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            lines = [
                f"{now} - sc - DEBUG - debug msg",
                f"{now} - sc - INFO - info msg",
                f"{now} - sc - ERROR - error msg",
            ]
            self._make_log_file(tmp, "app.log", lines)
            lc = LogCollector(log_dirs=[tmp])
            entries = lc.collect(since_hours=1, min_level=LogLevel.ERROR)
            assert len(entries) == 1
            assert entries[0].level == LogLevel.ERROR

    def test_collect_with_component_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            lines = [
                f"{now} - sc - INFO - SC message",
                f"{now} - mqtt - INFO - MQTT message",
            ]
            self._make_log_file(tmp, "app.log", lines)
            lc = LogCollector(log_dirs=[tmp])
            entries = lc.collect(since_hours=1, component_filter="sc")
            assert len(entries) == 1
            assert entries[0].component == "sc"

    def test_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            lines = [
                f"{now} - sc - INFO - Server started on port 8000",
                f"{now} - sc - ERROR - Connection timeout",
                f"{now} - mqtt - INFO - Broker ready",
            ]
            self._make_log_file(tmp, "app.log", lines)
            lc = LogCollector(log_dirs=[tmp])
            results = lc.search("timeout", since_hours=1)
            assert len(results) == 1
            assert "timeout" in results[0].message.lower()

    def test_error_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            lines = [
                f"{now} - sc - ERROR - err1",
                f"{now} - sc - ERROR - err2",
                f"{now} - mqtt - ERROR - err3",
                f"{now} - sc - INFO - ok",
            ]
            self._make_log_file(tmp, "app.log", lines)
            lc = LogCollector(log_dirs=[tmp])
            summary = lc.error_summary(since_hours=1)
            assert summary.get("sc", 0) == 2
            assert summary.get("mqtt", 0) == 1

    def test_log_entry_to_dict(self):
        entry = LogEntry(
            timestamp=1000.0,
            level=LogLevel.WARNING,
            component="edge",
            message="Low battery",
            source_file="/var/log/edge.log",
            line_number=42,
        )
        d = entry.to_dict()
        assert d["level"] == "WARNING"
        assert d["component"] == "edge"
        assert d["line_number"] == 42

    def test_log_entry_level_order(self):
        debug = LogEntry(level=LogLevel.DEBUG)
        error = LogEntry(level=LogLevel.ERROR)
        critical = LogEntry(level=LogLevel.CRITICAL)
        assert debug.level_order < error.level_order
        assert error.level_order < critical.level_order

    def test_empty_log_dirs(self):
        lc = LogCollector(log_dirs=[])
        assert lc.find_log_files() == []
        assert lc.collect() == []

    def test_nonexistent_log_dir(self):
        lc = LogCollector(log_dirs=["/nonexistent/log/dir"])
        assert lc.find_log_files() == []


# ---------------------------------------------------------------------------
# Integration / Smoke
# ---------------------------------------------------------------------------

class TestDeploymentIntegration:
    """Integration tests combining multiple deployment components."""

    def test_full_workflow(self):
        """End-to-end: config -> requirements -> health -> backup -> logs."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            log_dir = os.path.join(tmp, "logs")
            backup_dir = os.path.join(tmp, "backups")
            pid_dir = os.path.join(tmp, "pids")
            os.makedirs(data_dir)
            os.makedirs(log_dir)
            os.makedirs(pid_dir)

            # Write some data
            with open(os.path.join(data_dir, "targets.db"), "w") as f:
                f.write("target data")

            # Write a PID file for SC
            with open(os.path.join(pid_dir, "sc.pid"), "w") as f:
                f.write(str(os.getpid()))

            # Write a log file
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(log_dir, "sc.log"), "w") as f:
                f.write(f"{now} - sc - INFO - Server started\n")

            # 1. Create config
            cfg = DeploymentConfig(
                name="integration-test",
                components=["sc"],
                data_dir=data_dir,
                log_dir=log_dir,
            )
            assert cfg.validate() == []

            # 2. Check requirements
            reqs = SystemRequirements(min_python=(3, 10), min_disk_gb=0.001)
            result = reqs.check_local()
            assert result.meets_minimum is True

            # 3. Health check
            hc = HealthCheck(cfg, pid_dir=pid_dir)
            report = hc.check_all()
            assert report.overall == "healthy"
            assert hc.check_data_dir() is True
            assert hc.check_log_dir() is True

            # 4. Backup
            bm = BackupManager(backup_dir=backup_dir)
            manifest = bm.create_backup(data_dir, name="pre-update")
            assert manifest.file_count == 1

            # 5. Restore
            restore_dir = os.path.join(tmp, "restored")
            bm.restore_backup(manifest.backup_id, restore_dir)
            assert os.path.isfile(os.path.join(restore_dir, "targets.db"))

            # 6. Collect logs
            lc = LogCollector(log_dirs=[log_dir])
            entries = lc.collect(since_hours=1)
            assert len(entries) >= 1
            assert entries[0].component == "sc"
