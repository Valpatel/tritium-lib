# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.debug.streams — DebugStream and DebugOverlay."""

import pytest

from tritium_lib.sim_engine.debug.streams import (
    DebugStream,
    DebugFrame,
    DebugOverlay,
)


class TestDebugFrame:
    def test_construction(self):
        f = DebugFrame(system="test", tick=1, timestamp=100.0)
        assert f.system == "test"
        assert f.tick == 1
        assert f.entries == []

    def test_entries_mutable(self):
        f = DebugFrame(system="test", tick=1, timestamp=100.0)
        f.entries.append({"type": "data", "value": 42})
        assert len(f.entries) == 1


class TestDebugStream:
    def test_default_disabled(self):
        ds = DebugStream("test")
        assert not ds.enabled
        assert ds.system == "test"

    def test_begin_frame_when_disabled_returns_none(self):
        ds = DebugStream("test")
        assert ds.begin_frame() is None

    def test_begin_frame_when_enabled(self):
        ds = DebugStream("test")
        ds.enabled = True
        frame = ds.begin_frame()
        assert frame is not None
        assert frame.system == "test"
        assert frame.tick == 1

    def test_end_frame_stores_frame(self):
        ds = DebugStream("test")
        ds.enabled = True
        frame = ds.begin_frame()
        frame.entries.append({"a": 1})
        ds.end_frame(frame)
        assert ds.latest is frame
        assert len(ds.history) == 1

    def test_end_frame_none_is_safe(self):
        ds = DebugStream("test")
        ds.end_frame(None)  # Should not raise

    def test_tick_increments(self):
        ds = DebugStream("test")
        ds.enabled = True
        f1 = ds.begin_frame()
        ds.end_frame(f1)
        f2 = ds.begin_frame()
        ds.end_frame(f2)
        assert f2.tick == 2

    def test_max_history_limit(self):
        ds = DebugStream("test", max_history=3)
        ds.enabled = True
        for i in range(10):
            f = ds.begin_frame()
            ds.end_frame(f)
        assert len(ds.history) == 3

    def test_latest_returns_most_recent(self):
        ds = DebugStream("test")
        ds.enabled = True
        for i in range(5):
            f = ds.begin_frame()
            f.entries.append({"index": i})
            ds.end_frame(f)
        assert ds.latest.entries[0]["index"] == 4

    def test_latest_empty_returns_none(self):
        ds = DebugStream("test")
        assert ds.latest is None

    def test_listener_called_on_frame(self):
        ds = DebugStream("test")
        ds.enabled = True
        received = []
        ds.on_frame(lambda f: received.append(f))
        frame = ds.begin_frame()
        ds.end_frame(frame)
        assert len(received) == 1
        assert received[0] is frame

    def test_multiple_listeners(self):
        ds = DebugStream("test")
        ds.enabled = True
        count = [0]
        ds.on_frame(lambda f: count.__setitem__(0, count[0] + 1))
        ds.on_frame(lambda f: count.__setitem__(0, count[0] + 1))
        frame = ds.begin_frame()
        ds.end_frame(frame)
        assert count[0] == 2


class TestDebugOverlay:
    def test_empty_overlay(self):
        overlay = DebugOverlay()
        assert overlay.streams == {}
        assert overlay.get_snapshot() == {}

    def test_register_stream(self):
        overlay = DebugOverlay()
        ds = DebugStream("physics")
        overlay.register(ds)
        assert "physics" in overlay.streams

    def test_enable_all(self):
        overlay = DebugOverlay()
        ds1 = DebugStream("physics")
        ds2 = DebugStream("ai")
        overlay.register(ds1)
        overlay.register(ds2)
        overlay.enable_all()
        assert ds1.enabled
        assert ds2.enabled

    def test_disable_all(self):
        overlay = DebugOverlay()
        ds1 = DebugStream("physics")
        ds1.enabled = True
        overlay.register(ds1)
        overlay.disable_all()
        assert not ds1.enabled

    def test_get_snapshot(self):
        overlay = DebugOverlay()
        ds = DebugStream("test")
        ds.enabled = True
        overlay.register(ds)
        frame = ds.begin_frame()
        frame.entries.append({"data": 42})
        ds.end_frame(frame)
        snap = overlay.get_snapshot()
        assert "test" in snap
        assert snap["test"] is frame

    def test_to_dict(self):
        overlay = DebugOverlay()
        ds = DebugStream("combat")
        ds.enabled = True
        overlay.register(ds)
        frame = ds.begin_frame()
        frame.entries.append({"hit": True})
        ds.end_frame(frame)
        d = overlay.to_dict()
        assert "combat" in d
        assert d["combat"]["tick"] == 1
        assert d["combat"]["entry_count"] == 1

    def test_to_dict_empty_stream_excluded(self):
        overlay = DebugOverlay()
        ds = DebugStream("empty")
        overlay.register(ds)
        d = overlay.to_dict()
        assert "empty" not in d
