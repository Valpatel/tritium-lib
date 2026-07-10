# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the JSONL recording disk-retention sweep.

Extracted from tritium-sc (tests/engine/api/test_sim_recordings_retention.py)
alongside the ``sweep_recordings`` implementation.  Pins the durable
retention bound: delete oldest-first by mtime beyond a retention age
and/or a total-size cap, never touch files outside the recordings dir,
never delete non-JSONL files, and treat a missing directory as a no-op.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tritium_lib.recording import sweep_recordings

pytestmark = pytest.mark.unit


def _make_recording(directory: Path, name: str, size: int, age_days: float) -> Path:
    path = directory / f"{name}.jsonl"
    path.write_bytes(b"x" * size)
    mtime = time.time() - age_days * 86400.0
    os.utime(path, (mtime, mtime))
    return path


class TestSweepRecordings:
    def test_age_based_eviction(self, tmp_path):
        old = _make_recording(tmp_path, "old_battle", 10, age_days=30)
        recent = _make_recording(tmp_path, "recent_battle", 10, age_days=1)

        result = sweep_recordings(
            tmp_path, retention_days=7, max_total_bytes=0
        )

        assert not old.exists()
        assert recent.exists()
        assert "old_battle.jsonl" in result["removed"]
        assert result["freed_bytes"] == 10

    def test_size_based_eviction_oldest_first(self, tmp_path):
        # Three 100-byte files, ages 5/3/1 days; cap at 150 bytes forces
        # the two oldest out, leaving the newest (under cap).
        f5 = _make_recording(tmp_path, "age5", 100, age_days=5)
        f3 = _make_recording(tmp_path, "age3", 100, age_days=3)
        f1 = _make_recording(tmp_path, "age1", 100, age_days=1)

        result = sweep_recordings(
            tmp_path, retention_days=0, max_total_bytes=150
        )

        assert not f5.exists()
        assert not f3.exists()
        assert f1.exists()
        assert result["remaining_bytes"] <= 150

    def test_no_op_when_under_limits(self, tmp_path):
        f = _make_recording(tmp_path, "fresh", 10, age_days=0.1)
        result = sweep_recordings(
            tmp_path, retention_days=7, max_total_bytes=10_000
        )
        assert f.exists()
        assert result["removed"] == []
        assert result["freed_bytes"] == 0

    def test_ignores_non_jsonl_files(self, tmp_path):
        keep_txt = tmp_path / "notes.txt"
        keep_txt.write_bytes(b"y" * 1000)
        os.utime(keep_txt, (0, 0))  # ancient
        old_jsonl = _make_recording(tmp_path, "old", 10, age_days=30)

        sweep_recordings(tmp_path, retention_days=7, max_total_bytes=0)

        assert keep_txt.exists()  # never touched
        assert not old_jsonl.exists()

    def test_missing_dir_is_safe(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        result = sweep_recordings(missing, retention_days=7, max_total_bytes=0)
        assert result["removed"] == []
        assert result["freed_bytes"] == 0

    def test_string_directory_accepted(self, tmp_path):
        old = _make_recording(tmp_path, "old", 10, age_days=30)
        result = sweep_recordings(
            str(tmp_path), retention_days=7, max_total_bytes=0
        )
        assert not old.exists()
        assert result["removed"] == ["old.jsonl"]

    def test_subdirectories_never_touched(self, tmp_path):
        sub = tmp_path / "nested"
        sub.mkdir()
        nested = _make_recording(sub, "nested_old", 10, age_days=30)

        result = sweep_recordings(tmp_path, retention_days=7, max_total_bytes=0)

        assert nested.exists()
        assert result["removed"] == []
