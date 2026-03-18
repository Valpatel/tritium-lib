# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""Subprocess manager for Tritium addons — tracks all subprocesses per addon, prevents orphans."""

from __future__ import annotations

import signal
import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class ManagedProcess:
    """A tracked subprocess spawned by an addon."""

    key: str
    process: subprocess.Popen
    device_id: str
    process_name: str
    started_at: float
    command: list[str]

    @property
    def is_running(self) -> bool:
        """Whether the process is still running."""
        return self.process.poll() is None

    @property
    def pid(self) -> int:
        """The process ID."""
        return self.process.pid

    @property
    def runtime(self) -> float:
        """Seconds since the process was started."""
        return time.time() - self.started_at


class SubprocessManager:
    """Tracks all subprocesses for an addon, prevents orphans.

    Usage::

        mgr = SubprocessManager("my_addon")
        proc = mgr.spawn("device_1", "ffmpeg", ["ffmpeg", "-i", "rtsp://..."])
        mgr.kill_all()  # cleanup on shutdown
    """

    def __init__(self, addon_id: str) -> None:
        self.addon_id = addon_id
        self._processes: dict[str, ManagedProcess] = {}

    @staticmethod
    def _make_key(device_id: str, process_name: str) -> str:
        """Build the dict key for a process."""
        return f"{device_id}:{process_name}"

    def spawn(
        self,
        device_id: str,
        process_name: str,
        command: list[str],
        **popen_kwargs,
    ) -> ManagedProcess:
        """Start a subprocess and track it.

        If a process with the same key already exists and is running,
        it will be killed first.
        """
        key = self._make_key(device_id, process_name)

        # Kill existing process with the same key if running
        existing = self._processes.get(key)
        if existing is not None and existing.is_running:
            self.kill(device_id, process_name)

        proc = subprocess.Popen(command, **popen_kwargs)
        managed = ManagedProcess(
            key=key,
            process=proc,
            device_id=device_id,
            process_name=process_name,
            started_at=time.time(),
            command=command,
        )
        self._processes[key] = managed
        return managed

    def kill(self, device_id: str, process_name: str, timeout: float = 5.0) -> bool:
        """Kill a specific process gracefully (SIGTERM then SIGKILL).

        Returns True if a process was found and killed, False otherwise.
        """
        key = self._make_key(device_id, process_name)
        managed = self._processes.get(key)
        if managed is None:
            return False

        if managed.is_running:
            try:
                managed.process.terminate()  # SIGTERM
                try:
                    managed.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    managed.process.kill()  # SIGKILL
                    managed.process.wait(timeout=2.0)
            except OSError:
                pass  # Process already gone

        del self._processes[key]
        return True

    def kill_device(self, device_id: str) -> int:
        """Kill all processes for a given device. Returns count killed."""
        keys = [k for k, v in self._processes.items() if v.device_id == device_id]
        count = 0
        for key in keys:
            managed = self._processes[key]
            if managed.is_running:
                try:
                    managed.process.terminate()
                    try:
                        managed.process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        managed.process.kill()
                        managed.process.wait(timeout=2.0)
                except OSError:
                    pass
            del self._processes[key]
            count += 1
        return count

    def kill_all(self, timeout: float = 5.0) -> int:
        """Kill all tracked processes. Returns count killed."""
        count = len(self._processes)
        for managed in list(self._processes.values()):
            if managed.is_running:
                try:
                    managed.process.terminate()
                    try:
                        managed.process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        managed.process.kill()
                        managed.process.wait(timeout=2.0)
                except OSError:
                    pass
        self._processes.clear()
        return count

    def get(self, device_id: str, process_name: str) -> ManagedProcess | None:
        """Get a tracked process by device and name."""
        return self._processes.get(self._make_key(device_id, process_name))

    def list_processes(self) -> list[ManagedProcess]:
        """List all tracked processes (running and dead)."""
        return list(self._processes.values())

    def list_running(self) -> list[ManagedProcess]:
        """List only currently running processes."""
        return [m for m in self._processes.values() if m.is_running]

    def is_running(self, device_id: str, process_name: str) -> bool:
        """Check if a specific process is running."""
        managed = self.get(device_id, process_name)
        return managed is not None and managed.is_running

    def cleanup_dead(self) -> int:
        """Remove finished processes from tracking. Returns count removed."""
        dead_keys = [k for k, v in self._processes.items() if not v.is_running]
        for key in dead_keys:
            del self._processes[key]
        return len(dead_keys)

    @property
    def process_count(self) -> int:
        """Total number of tracked processes."""
        return len(self._processes)

    @property
    def running_count(self) -> int:
        """Number of currently running processes."""
        return sum(1 for v in self._processes.values() if v.is_running)

    def to_dict(self) -> dict:
        """JSON-serializable status of the manager."""
        return {
            "addon_id": self.addon_id,
            "process_count": self.process_count,
            "running_count": self.running_count,
            "processes": [
                {
                    "key": m.key,
                    "device_id": m.device_id,
                    "process_name": m.process_name,
                    "pid": m.pid,
                    "is_running": m.is_running,
                    "runtime": round(m.runtime, 2),
                    "command": m.command,
                }
                for m in self._processes.values()
            ],
        }
