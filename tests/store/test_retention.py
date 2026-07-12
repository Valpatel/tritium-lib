# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the shared disk-retention core (tritium_lib.store.retention).

The behavioral twins — recording ``sweep_recordings`` and
``GISCache.sweep`` — each keep their own suites; this file pins the
shared core's own contract, especially the knobs the wrappers differ
on (recursive vs flat, suffix allowlist, empty-dir pruning).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tritium_lib.store.retention import sweep_dir

pytestmark = pytest.mark.unit


def _mk(directory: Path, rel: str, size: int, age_days: float) -> Path:
    path = directory / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    mtime = time.time() - age_days * 86400.0
    os.utime(path, (mtime, mtime))
    return path


class TestSweepDirCore:
    def test_age_then_size(self, tmp_path):
        old = _mk(tmp_path, "old.jsonl", 10, 30)
        mid = _mk(tmp_path, "mid.jsonl", 100, 3)
        new = _mk(tmp_path, "new.jsonl", 100, 1)

        result = sweep_dir(
            tmp_path, retention_days=7, max_total_bytes=150,
            suffixes=(".jsonl",),
        )

        assert not old.exists()      # age pass
        assert not mid.exists()      # size pass, oldest survivor first
        assert new.exists()
        assert result["freed_bytes"] == 110
        assert result["remaining_bytes"] == 100

    def test_flat_mode_ignores_nested(self, tmp_path):
        nested = _mk(tmp_path, "sub/nested.jsonl", 10, 30)
        top = _mk(tmp_path, "top.jsonl", 10, 30)

        result = sweep_dir(
            tmp_path, retention_days=7, max_total_bytes=0,
            suffixes=(".jsonl",), recursive=False,
        )

        assert nested.exists()
        assert not top.exists()
        assert result["removed"] == ["top.jsonl"]

    def test_recursive_mode_sweeps_nested_and_prunes(self, tmp_path):
        nested = _mk(tmp_path, "tiles/1/2/3.json", 10, 30)
        result = sweep_dir(
            tmp_path, retention_days=7, max_total_bytes=0,
            suffixes=(".json", ".bin"), recursive=True, prune_empty=True,
        )
        assert not nested.exists()
        assert result["removed"] == [str(Path("tiles/1/2/3.json"))]
        assert not (tmp_path / "tiles").exists()  # empty tree pruned

    def test_suffix_allowlist_is_hard(self, tmp_path):
        note = _mk(tmp_path, "note.txt", 10, 365)
        lock = _mk(tmp_path, "store.lock", 10, 365)
        data = _mk(tmp_path, "data.json", 10, 365)

        sweep_dir(
            tmp_path, retention_days=7, max_total_bytes=0,
            suffixes=(".json",),
        )

        assert note.exists()
        assert lock.exists()
        assert not data.exists()

    def test_out_of_tree_symlink_never_deleted(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        victim = _mk(outside, "victim.json", 10, 365)
        store = tmp_path / "store"
        store.mkdir()
        link = store / "link.json"
        link.symlink_to(victim)
        mtime = time.time() - 365 * 86400.0
        os.utime(link, (mtime, mtime), follow_symlinks=False)

        sweep_dir(
            store, retention_days=7, max_total_bytes=0, suffixes=(".json",),
        )

        assert victim.exists(), "symlink target outside the tree must survive"

    def test_missing_dir_noop(self, tmp_path):
        result = sweep_dir(
            tmp_path / "nope", retention_days=7, max_total_bytes=0,
            suffixes=(".json",),
        )
        assert result == {"removed": [], "freed_bytes": 0, "remaining_bytes": 0}

    def test_both_bounds_disabled_keeps_everything(self, tmp_path):
        f = _mk(tmp_path, "ancient.json", 10, 3650)
        result = sweep_dir(
            tmp_path, retention_days=0, max_total_bytes=0, suffixes=(".json",),
        )
        assert f.exists()
        assert result["removed"] == []
        assert result["remaining_bytes"] == 10
