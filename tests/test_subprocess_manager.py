# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""Tests for SubprocessManager — addon subprocess lifecycle tracking."""

import time

import pytest

from tritium_lib.sdk.subprocess_manager import ManagedProcess, SubprocessManager


@pytest.fixture
def mgr():
    """Create a SubprocessManager and ensure all processes are cleaned up."""
    m = SubprocessManager("test_addon")
    yield m
    m.kill_all(timeout=2.0)


class TestManagedProcess:
    def test_properties(self, mgr: SubprocessManager):
        proc = mgr.spawn("dev1", "sleeper", ["sleep", "10"])
        assert proc.is_running
        assert proc.pid > 0
        assert proc.runtime >= 0
        assert proc.key == "dev1:sleeper"
        assert proc.device_id == "dev1"
        assert proc.process_name == "sleeper"
        assert proc.command == ["sleep", "10"]


class TestSpawn:
    def test_spawn_and_track(self, mgr: SubprocessManager):
        proc = mgr.spawn("dev1", "worker", ["sleep", "10"])
        assert mgr.process_count == 1
        assert mgr.running_count == 1
        assert mgr.is_running("dev1", "worker")

    def test_spawn_replaces_existing(self, mgr: SubprocessManager):
        proc1 = mgr.spawn("dev1", "worker", ["sleep", "10"])
        pid1 = proc1.pid
        proc2 = mgr.spawn("dev1", "worker", ["sleep", "10"])
        pid2 = proc2.pid
        assert pid1 != pid2
        assert mgr.process_count == 1
        # Old process should be dead
        assert proc1.process.poll() is not None

    def test_spawn_multiple_devices(self, mgr: SubprocessManager):
        mgr.spawn("dev1", "worker", ["sleep", "10"])
        mgr.spawn("dev2", "worker", ["sleep", "10"])
        assert mgr.process_count == 2

    def test_spawn_multiple_names(self, mgr: SubprocessManager):
        mgr.spawn("dev1", "ffmpeg", ["sleep", "10"])
        mgr.spawn("dev1", "rtsp", ["sleep", "10"])
        assert mgr.process_count == 2


class TestKill:
    def test_kill_running(self, mgr: SubprocessManager):
        mgr.spawn("dev1", "worker", ["sleep", "10"])
        assert mgr.kill("dev1", "worker")
        assert mgr.process_count == 0
        assert not mgr.is_running("dev1", "worker")

    def test_kill_nonexistent(self, mgr: SubprocessManager):
        assert not mgr.kill("dev1", "worker")

    def test_kill_already_dead(self, mgr: SubprocessManager):
        proc = mgr.spawn("dev1", "quick", ["true"])
        proc.process.wait()  # Wait for it to finish
        assert mgr.kill("dev1", "quick")
        assert mgr.process_count == 0


class TestKillDevice:
    def test_kill_device(self, mgr: SubprocessManager):
        mgr.spawn("dev1", "a", ["sleep", "10"])
        mgr.spawn("dev1", "b", ["sleep", "10"])
        mgr.spawn("dev2", "a", ["sleep", "10"])
        count = mgr.kill_device("dev1")
        assert count == 2
        assert mgr.process_count == 1
        assert not mgr.is_running("dev1", "a")
        assert not mgr.is_running("dev1", "b")
        assert mgr.is_running("dev2", "a")

    def test_kill_device_empty(self, mgr: SubprocessManager):
        assert mgr.kill_device("nonexistent") == 0


class TestKillAll:
    def test_kill_all(self, mgr: SubprocessManager):
        mgr.spawn("dev1", "a", ["sleep", "10"])
        mgr.spawn("dev2", "b", ["sleep", "10"])
        mgr.spawn("dev3", "c", ["sleep", "10"])
        count = mgr.kill_all()
        assert count == 3
        assert mgr.process_count == 0
        assert mgr.running_count == 0

    def test_kill_all_empty(self, mgr: SubprocessManager):
        assert mgr.kill_all() == 0


class TestGet:
    def test_get_existing(self, mgr: SubprocessManager):
        mgr.spawn("dev1", "worker", ["sleep", "10"])
        proc = mgr.get("dev1", "worker")
        assert proc is not None
        assert proc.device_id == "dev1"

    def test_get_nonexistent(self, mgr: SubprocessManager):
        assert mgr.get("dev1", "worker") is None


class TestListProcesses:
    def test_list_processes(self, mgr: SubprocessManager):
        mgr.spawn("dev1", "a", ["sleep", "10"])
        mgr.spawn("dev2", "b", ["sleep", "10"])
        procs = mgr.list_processes()
        assert len(procs) == 2

    def test_list_running(self, mgr: SubprocessManager):
        mgr.spawn("dev1", "running", ["sleep", "10"])
        quick = mgr.spawn("dev1", "done", ["true"])
        quick.process.wait()
        running = mgr.list_running()
        assert len(running) == 1
        assert running[0].process_name == "running"


class TestCleanupDead:
    def test_cleanup_dead(self, mgr: SubprocessManager):
        mgr.spawn("dev1", "alive", ["sleep", "10"])
        quick = mgr.spawn("dev1", "dead", ["true"])
        quick.process.wait()
        removed = mgr.cleanup_dead()
        assert removed == 1
        assert mgr.process_count == 1
        assert mgr.is_running("dev1", "alive")

    def test_cleanup_dead_none(self, mgr: SubprocessManager):
        mgr.spawn("dev1", "alive", ["sleep", "10"])
        assert mgr.cleanup_dead() == 0


class TestToDict:
    def test_to_dict(self, mgr: SubprocessManager):
        mgr.spawn("dev1", "worker", ["sleep", "10"])
        d = mgr.to_dict()
        assert d["addon_id"] == "test_addon"
        assert d["process_count"] == 1
        assert d["running_count"] == 1
        assert len(d["processes"]) == 1
        p = d["processes"][0]
        assert p["key"] == "dev1:worker"
        assert p["device_id"] == "dev1"
        assert p["process_name"] == "worker"
        assert p["is_running"] is True
        assert isinstance(p["pid"], int)
        assert isinstance(p["runtime"], float)
        assert p["command"] == ["sleep", "10"]

    def test_to_dict_empty(self, mgr: SubprocessManager):
        d = mgr.to_dict()
        assert d["process_count"] == 0
        assert d["processes"] == []


class TestProperties:
    def test_process_count(self, mgr: SubprocessManager):
        assert mgr.process_count == 0
        mgr.spawn("dev1", "a", ["sleep", "10"])
        assert mgr.process_count == 1
        mgr.spawn("dev2", "b", ["sleep", "10"])
        assert mgr.process_count == 2

    def test_running_count_with_dead(self, mgr: SubprocessManager):
        mgr.spawn("dev1", "alive", ["sleep", "10"])
        quick = mgr.spawn("dev1", "dead", ["true"])
        quick.process.wait()
        assert mgr.process_count == 2
        assert mgr.running_count == 1
