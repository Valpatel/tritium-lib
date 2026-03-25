# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for SubprocessManager — spawn, kill, cleanup, status."""

import subprocess
import sys
import time

import pytest

from tritium_lib.sdk.subprocess_manager import ManagedProcess, SubprocessManager


@pytest.fixture
def mgr():
    """Create a SubprocessManager and ensure cleanup."""
    m = SubprocessManager("test_addon")
    yield m
    m.kill_all(timeout=2)


def _sleep_cmd(seconds=10):
    """Return a command that sleeps for N seconds (cross-platform)."""
    return [sys.executable, "-c", f"import time; time.sleep({seconds})"]


# ── ManagedProcess ──────────────────────────────────────────────────

class TestManagedProcess:
    """Tests for the ManagedProcess dataclass."""

    def test_is_running_true(self, mgr):
        proc = mgr.spawn("dev1", "sleeper", _sleep_cmd(10))
        assert proc.is_running is True

    def test_is_running_false_after_exit(self, mgr):
        proc = mgr.spawn("dev1", "quick", [sys.executable, "-c", "pass"])
        proc.process.wait(timeout=5)
        assert proc.is_running is False

    def test_pid_is_positive(self, mgr):
        proc = mgr.spawn("dev1", "sleeper", _sleep_cmd(10))
        assert proc.pid > 0

    def test_runtime_increases(self, mgr):
        proc = mgr.spawn("dev1", "sleeper", _sleep_cmd(10))
        r1 = proc.runtime
        time.sleep(0.05)
        r2 = proc.runtime
        assert r2 > r1

    def test_key_format(self, mgr):
        proc = mgr.spawn("device-1", "ffmpeg", _sleep_cmd(10))
        assert proc.key == "device-1:ffmpeg"

    def test_command_stored(self, mgr):
        cmd = _sleep_cmd(10)
        proc = mgr.spawn("dev1", "proc1", cmd)
        assert proc.command == cmd


# ── SubprocessManager spawn/kill ────────────────────────────────────

class TestSubprocessManagerSpawn:
    """Tests for spawn and kill operations."""

    def test_spawn_creates_process(self, mgr):
        proc = mgr.spawn("dev1", "worker", _sleep_cmd(10))
        assert proc is not None
        assert proc.is_running is True

    def test_spawn_replaces_existing(self, mgr):
        p1 = mgr.spawn("dev1", "worker", _sleep_cmd(10))
        pid1 = p1.pid
        p2 = mgr.spawn("dev1", "worker", _sleep_cmd(10))
        assert p2.pid != pid1
        assert mgr.process_count == 1

    def test_kill_specific_process(self, mgr):
        mgr.spawn("dev1", "worker", _sleep_cmd(10))
        assert mgr.kill("dev1", "worker") is True
        assert mgr.process_count == 0

    def test_kill_nonexistent_returns_false(self, mgr):
        assert mgr.kill("dev1", "worker") is False

    def test_kill_device(self, mgr):
        mgr.spawn("dev1", "worker1", _sleep_cmd(10))
        mgr.spawn("dev1", "worker2", _sleep_cmd(10))
        mgr.spawn("dev2", "worker1", _sleep_cmd(10))
        count = mgr.kill_device("dev1")
        assert count == 2
        assert mgr.process_count == 1

    def test_kill_all(self, mgr):
        mgr.spawn("dev1", "w1", _sleep_cmd(10))
        mgr.spawn("dev2", "w2", _sleep_cmd(10))
        count = mgr.kill_all()
        assert count == 2
        assert mgr.process_count == 0


# ── SubprocessManager queries ───────────────────────────────────────

class TestSubprocessManagerQueries:
    """Tests for process queries and status."""

    def test_get_existing(self, mgr):
        mgr.spawn("dev1", "worker", _sleep_cmd(10))
        proc = mgr.get("dev1", "worker")
        assert proc is not None
        assert proc.device_id == "dev1"

    def test_get_nonexistent(self, mgr):
        assert mgr.get("dev1", "worker") is None

    def test_list_processes(self, mgr):
        mgr.spawn("dev1", "w1", _sleep_cmd(10))
        mgr.spawn("dev2", "w2", _sleep_cmd(10))
        procs = mgr.list_processes()
        assert len(procs) == 2

    def test_list_running(self, mgr):
        mgr.spawn("dev1", "long", _sleep_cmd(10))
        mgr.spawn("dev2", "short", [sys.executable, "-c", "pass"])
        # Wait for the short one to finish
        time.sleep(0.5)
        running = mgr.list_running()
        assert len(running) >= 1  # At least the long one

    def test_is_running(self, mgr):
        mgr.spawn("dev1", "worker", _sleep_cmd(10))
        assert mgr.is_running("dev1", "worker") is True
        assert mgr.is_running("dev1", "nonexistent") is False

    def test_process_count(self, mgr):
        assert mgr.process_count == 0
        mgr.spawn("dev1", "w1", _sleep_cmd(10))
        assert mgr.process_count == 1
        mgr.spawn("dev2", "w2", _sleep_cmd(10))
        assert mgr.process_count == 2

    def test_running_count(self, mgr):
        mgr.spawn("dev1", "long", _sleep_cmd(10))
        assert mgr.running_count >= 1

    def test_cleanup_dead(self, mgr):
        mgr.spawn("dev1", "short", [sys.executable, "-c", "pass"])
        time.sleep(0.5)
        removed = mgr.cleanup_dead()
        assert removed >= 1
        assert mgr.process_count == 0

    def test_to_dict(self, mgr):
        mgr.spawn("dev1", "worker", _sleep_cmd(10))
        d = mgr.to_dict()
        assert d["addon_id"] == "test_addon"
        assert d["process_count"] == 1
        assert d["running_count"] >= 1
        assert len(d["processes"]) == 1
        p = d["processes"][0]
        assert p["device_id"] == "dev1"
        assert p["process_name"] == "worker"
        assert "pid" in p
        assert "is_running" in p
        assert "runtime" in p
        assert "command" in p

    def test_addon_id_stored(self, mgr):
        assert mgr.addon_id == "test_addon"
