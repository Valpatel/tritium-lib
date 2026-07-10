# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for GISCache.sweep — age/size retention mirroring sweep_recordings.

No network.  Files are laid down in a tmp cache dir with controlled mtimes
(``os.utime`` against an explicit ``now``) so the two eviction passes, the
suffix allowlist, symlink safety, empty-dir pruning, and the missing-dir no-op
are all deterministic.
"""

import os
from pathlib import Path

import pytest

from tritium_lib.geo.gis.cache import GISCache

DAY = 86400.0
NOW = 10_000_000.0


def _write(path: Path, size: int, mtime: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))
    return path


class TestAgePass:
    @pytest.mark.unit
    def test_old_removed_fresh_kept(self, tmp_path):
        c = GISCache(tmp_path)
        old = _write(tmp_path / "old.json", 100, NOW - 40 * DAY)
        fresh = _write(tmp_path / "fresh.json", 100, NOW - 1 * DAY)
        res = c.sweep(retention_days=30, max_total_bytes=0, now=NOW)
        assert res["removed"] == ["old.json"]
        assert res["freed_bytes"] == 100
        assert res["remaining_bytes"] == 100
        assert not old.exists()
        assert fresh.exists()

    @pytest.mark.unit
    def test_retention_zero_skips_age_pass(self, tmp_path):
        c = GISCache(tmp_path)
        _write(tmp_path / "ancient.json", 50, NOW - 999 * DAY)
        res = c.sweep(retention_days=0, max_total_bytes=0, now=NOW)
        assert res["removed"] == []
        assert (tmp_path / "ancient.json").exists()


class TestSizePass:
    @pytest.mark.unit
    def test_size_evicts_oldest_first(self, tmp_path):
        c = GISCache(tmp_path)
        _write(tmp_path / "a.json", 100, NOW - 3 * DAY)  # oldest
        _write(tmp_path / "b.json", 100, NOW - 2 * DAY)
        _write(tmp_path / "c.json", 100, NOW - 1 * DAY)  # newest
        # total 300; cap 250 -> drop just the oldest.
        res = c.sweep(retention_days=0, max_total_bytes=250, now=NOW)
        assert res["removed"] == ["a.json"]
        assert res["remaining_bytes"] == 200
        assert not (tmp_path / "a.json").exists()
        assert (tmp_path / "b.json").exists()
        assert (tmp_path / "c.json").exists()

    @pytest.mark.unit
    def test_size_cap_zero_skips_size_pass(self, tmp_path):
        c = GISCache(tmp_path)
        _write(tmp_path / "big.json", 5000, NOW - 1 * DAY)
        res = c.sweep(retention_days=0, max_total_bytes=0, now=NOW)
        assert res["removed"] == []
        assert res["remaining_bytes"] == 5000


class TestAllowlist:
    @pytest.mark.unit
    def test_non_allowlisted_suffix_survives(self, tmp_path):
        c = GISCache(tmp_path)
        _write(tmp_path / "keep.txt", 100, NOW - 999 * DAY)
        _write(tmp_path / "drop.json", 100, NOW - 999 * DAY)
        _write(tmp_path / "tile.bin", 100, NOW - 999 * DAY)
        res = c.sweep(retention_days=1, max_total_bytes=0, now=NOW)
        assert set(res["removed"]) == {"drop.json", "tile.bin"}
        assert (tmp_path / "keep.txt").exists()  # never touched


class TestRecursiveAndPrune:
    @pytest.mark.unit
    def test_walks_tiles_tree_and_prunes_empty_dirs(self, tmp_path):
        c = GISCache(tmp_path)
        nested = tmp_path / "tiles" / "12" / "654"
        _write(nested / "1583.json", 100, NOW - 999 * DAY)
        res = c.sweep(retention_days=1, max_total_bytes=0, now=NOW)
        assert res["removed"] == [str(Path("tiles") / "12" / "654" / "1583.json")]
        assert not nested.exists()          # empty dirs pruned
        assert not (tmp_path / "tiles").exists()
        assert tmp_path.exists()            # base dir itself preserved


class TestSymlinkSafety:
    @pytest.mark.unit
    def test_out_of_tree_symlink_not_deleted(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = _write(outside / "secret.json", 100, NOW - 999 * DAY)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        link = cache_dir / "link.json"
        try:
            link.symlink_to(secret)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unsupported on this platform")

        c = GISCache(cache_dir)
        res = c.sweep(retention_days=1, max_total_bytes=0, now=NOW)
        # The symlink target resolves outside the cache tree -> never unlinked.
        assert "link.json" not in res["removed"]
        assert secret.exists()


class TestMissingDir:
    @pytest.mark.unit
    def test_missing_dir_is_noop(self, tmp_path):
        c = GISCache(tmp_path / "does-not-exist")
        res = c.sweep(retention_days=1, max_total_bytes=1, now=NOW)
        assert res == {"removed": [], "freed_bytes": 0, "remaining_bytes": 0}

    @pytest.mark.unit
    def test_empty_dir_is_noop(self, tmp_path):
        c = GISCache(tmp_path)
        res = c.sweep(retention_days=1, max_total_bytes=1, now=NOW)
        assert res == {"removed": [], "freed_bytes": 0, "remaining_bytes": 0}
