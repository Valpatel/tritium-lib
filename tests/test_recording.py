# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.recording — sensor data record/replay system."""

import json
import time
import pytest
from pathlib import Path

from tritium_lib.recording import Recorder, Player, ReplayEvent, Session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def recording_path(tmp_path: Path) -> Path:
    """Temp file path for recording output."""
    return tmp_path / "test_recording.jsonl"


@pytest.fixture
def populated_recording(recording_path: Path) -> Path:
    """Create a recording with known events and return its path."""
    rec = Recorder(recording_path, metadata={"mission": "test_patrol"})
    rec.start()
    rec.record("ble_sighting", source="node_alpha", data={"mac": "AA:BB:CC:DD:EE:FF", "rssi": -45}, timestamp=1000.0)
    rec.record("wifi_probe", source="node_alpha", data={"bssid": "00:11:22:33:44:55"}, timestamp=1001.0)
    rec.record("camera_detection", source="cam_01", data={"class": "person", "confidence": 0.92}, timestamp=1002.5)
    rec.record("ble_sighting", source="node_beta", data={"mac": "11:22:33:44:55:66", "rssi": -70}, timestamp=1003.0)
    rec.record("acoustic_event", source="mic_01", data={"class": "vehicle", "db": 72.3}, timestamp=1005.0)
    rec.record("fusion_result", source="pipeline", data={"target_id": "t_001", "sources": 3}, timestamp=1006.0)
    rec.record("alert", source="automation", data={"level": "warning", "message": "Unknown in zone"}, timestamp=1007.0)
    rec.stop()
    return recording_path


# ---------------------------------------------------------------------------
# Recorder tests
# ---------------------------------------------------------------------------


class TestRecorder:
    """Test the Recorder class."""

    def test_start_creates_file(self, recording_path: Path):
        rec = Recorder(recording_path)
        sid = rec.start()
        assert recording_path.exists()
        assert len(sid) == 32  # hex UUID
        rec.stop()

    def test_start_writes_header(self, recording_path: Path):
        rec = Recorder(recording_path, metadata={"site": "hq"})
        rec.start()
        rec.stop()
        with open(recording_path) as f:
            header = json.loads(f.readline())
        assert header["event_type"] == "_session_start"
        assert header["data"]["metadata"]["site"] == "hq"
        assert "session_id" in header["data"]

    def test_record_adds_events(self, recording_path: Path):
        rec = Recorder(recording_path)
        rec.start()
        rec.record("ble_sighting", source="node_a", data={"rssi": -55})
        rec.record("wifi_probe", source="node_b", data={"ssid": "TestNet"})
        assert rec.event_count == 2
        rec.stop()

    def test_record_custom_timestamp(self, recording_path: Path):
        rec = Recorder(recording_path)
        rec.start()
        rec.record("ble_sighting", timestamp=1234567890.5)
        rec.stop()
        with open(recording_path) as f:
            f.readline()  # skip header
            event = json.loads(f.readline())
        assert event["ts"] == 1234567890.5

    def test_stop_writes_footer(self, recording_path: Path):
        rec = Recorder(recording_path)
        rec.start()
        rec.record("ble_sighting", source="node_a")
        rec.record("camera_detection", source="cam_01")
        summary = rec.stop()
        assert summary["event_count"] == 2
        assert "ble_sighting" in summary["sensor_types"]
        assert "camera_detection" in summary["sensor_types"]
        assert "node_a" in summary["sources"]
        # Check footer in file
        with open(recording_path) as f:
            lines = f.readlines()
        footer = json.loads(lines[-1])
        assert footer["event_type"] == "_session_end"

    def test_stop_returns_duration(self, recording_path: Path):
        rec = Recorder(recording_path)
        rec.start()
        rec.record("ble_sighting", timestamp=100.0)
        summary = rec.stop()
        assert summary["duration"] >= 0
        assert "session_id" in summary

    def test_double_start_raises(self, recording_path: Path):
        rec = Recorder(recording_path)
        rec.start()
        with pytest.raises(RuntimeError, match="already active"):
            rec.start()
        rec.stop()

    def test_record_without_start_raises(self, recording_path: Path):
        rec = Recorder(recording_path)
        with pytest.raises(RuntimeError, match="not active"):
            rec.record("ble_sighting")

    def test_stop_without_start_raises(self, recording_path: Path):
        rec = Recorder(recording_path)
        with pytest.raises(RuntimeError, match="not active"):
            rec.stop()

    def test_context_manager(self, recording_path: Path):
        with Recorder(recording_path, metadata={"test": True}) as rec:
            assert rec.is_recording
            rec.record("ble_sighting", source="node_a")
        assert not rec.is_recording
        assert recording_path.exists()
        # File should have header, 1 event, footer = 3 lines
        with open(recording_path) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_creates_parent_directories(self, tmp_path: Path):
        deep_path = tmp_path / "a" / "b" / "c" / "rec.jsonl"
        rec = Recorder(deep_path)
        rec.start()
        rec.record("ble_sighting")
        rec.stop()
        assert deep_path.exists()

    def test_session_id_property(self, recording_path: Path):
        rec = Recorder(recording_path)
        assert rec.session_id == ""
        sid = rec.start()
        assert rec.session_id == sid
        rec.stop()


# ---------------------------------------------------------------------------
# Player tests
# ---------------------------------------------------------------------------


class TestPlayer:
    """Test the Player class."""

    def test_load_and_count(self, populated_recording: Path):
        player = Player(populated_recording)
        total = player.load()
        assert total == 9  # 7 events + header + footer
        assert player.event_count == 7  # skip_control=True by default

    def test_events_generator(self, populated_recording: Path):
        player = Player(populated_recording)
        events = list(player.events())
        assert len(events) == 7
        assert events[0].event_type == "ble_sighting"
        assert events[0].source == "node_alpha"
        assert events[0].data["mac"] == "AA:BB:CC:DD:EE:FF"

    def test_events_include_control(self, populated_recording: Path):
        player = Player(populated_recording, skip_control=False)
        events = list(player.events())
        assert len(events) == 9
        assert events[0].event_type == "_session_start"
        assert events[-1].event_type == "_session_end"

    def test_duration(self, populated_recording: Path):
        player = Player(populated_recording)
        # Header ts and footer ts span the full range
        assert player.duration > 0

    def test_sensor_types(self, populated_recording: Path):
        player = Player(populated_recording)
        types = player.sensor_types()
        assert "ble_sighting" in types
        assert "wifi_probe" in types
        assert "camera_detection" in types
        assert "acoustic_event" in types
        assert "fusion_result" in types
        assert "alert" in types
        assert "_session_start" not in types

    def test_sources(self, populated_recording: Path):
        player = Player(populated_recording)
        srcs = player.sources()
        assert "node_alpha" in srcs
        assert "node_beta" in srcs
        assert "cam_01" in srcs
        assert "mic_01" in srcs

    def test_slice_by_time(self, populated_recording: Path):
        player = Player(populated_recording)
        # Events at 1000, 1001, 1002.5, 1003, 1005, 1006, 1007
        events = player.slice(start_ts=1002.0, end_ts=1005.5)
        assert len(events) == 3  # 1002.5, 1003.0, 1005.0

    def test_slice_by_event_type(self, populated_recording: Path):
        player = Player(populated_recording)
        events = player.slice(event_types={"ble_sighting"})
        assert len(events) == 2
        assert all(e.event_type == "ble_sighting" for e in events)

    def test_slice_by_source(self, populated_recording: Path):
        player = Player(populated_recording)
        events = player.slice(sources={"node_alpha"})
        assert len(events) == 2  # ble_sighting + wifi_probe from node_alpha

    def test_slice_combined_filters(self, populated_recording: Path):
        player = Player(populated_recording)
        events = player.slice(
            start_ts=1000.0,
            end_ts=1003.0,
            event_types={"ble_sighting"},
            sources={"node_alpha"},
        )
        assert len(events) == 1
        assert events[0].data["mac"] == "AA:BB:CC:DD:EE:FF"

    def test_file_not_found(self, tmp_path: Path):
        player = Player(tmp_path / "nonexistent.jsonl")
        with pytest.raises(FileNotFoundError):
            player.load()

    def test_replay_event_to_dict(self):
        event = ReplayEvent(ts=1000.0, event_type="ble_sighting", source="node_a", data={"rssi": -50})
        d = event.to_dict()
        assert d["ts"] == 1000.0
        assert d["event_type"] == "ble_sighting"
        assert d["data"]["rssi"] == -50

    def test_replay_event_from_line(self):
        line = '{"ts":1000.0,"event_type":"wifi_probe","source":"node_b","data":{"ssid":"Test"}}'
        event = ReplayEvent.from_line(line)
        assert event.ts == 1000.0
        assert event.event_type == "wifi_probe"
        assert event.data["ssid"] == "Test"


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------


class TestSession:
    """Test the Session metadata class."""

    def test_from_complete_file(self, populated_recording: Path):
        session = Session.from_file(populated_recording)
        assert session.complete is True
        assert session.event_count == 7
        assert session.session_id != ""
        assert session.path == str(populated_recording)
        assert "ble_sighting" in session.sensor_types
        assert "node_alpha" in session.sources
        assert session.metadata["mission"] == "test_patrol"

    def test_from_incomplete_file(self, recording_path: Path):
        """Test session from a file without _session_end."""
        # Write raw events without footer
        with open(recording_path, "w") as f:
            f.write(json.dumps({"ts": 1000.0, "event_type": "ble_sighting", "source": "node_a", "data": {"rssi": -50}}) + "\n")
            f.write(json.dumps({"ts": 1002.0, "event_type": "wifi_probe", "source": "node_b", "data": {"ssid": "X"}}) + "\n")
        session = Session.from_file(recording_path)
        assert session.complete is False
        assert session.event_count == 2
        assert session.duration == pytest.approx(2.0)
        assert "ble_sighting" in session.sensor_types
        assert "wifi_probe" in session.sensor_types

    def test_empty_file_raises(self, recording_path: Path):
        recording_path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            Session.from_file(recording_path)

    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            Session.from_file(tmp_path / "nope.jsonl")

    def test_to_dict_roundtrip(self, populated_recording: Path):
        session = Session.from_file(populated_recording)
        d = session.to_dict()
        restored = Session.from_dict(d)
        assert restored.session_id == session.session_id
        assert restored.event_count == session.event_count
        assert restored.complete == session.complete
        assert restored.sensor_types == session.sensor_types
        assert restored.duration == pytest.approx(session.duration)

    def test_summary_string(self, populated_recording: Path):
        session = Session.from_file(populated_recording)
        s = session.summary()
        assert "complete" in s
        assert "7 events" in s


# ---------------------------------------------------------------------------
# Integration / round-trip tests
# ---------------------------------------------------------------------------


class TestRecordReplayRoundtrip:
    """Test full record -> replay cycle."""

    def test_record_then_replay_preserves_all_data(self, recording_path: Path):
        """Record events, replay them, verify every field matches."""
        events_in = [
            ("ble_sighting", "node_a", {"mac": "AA:BB:CC:DD:EE:FF", "rssi": -45}, 1000.0),
            ("wifi_probe", "node_b", {"bssid": "00:11:22:33:44:55"}, 1001.0),
            ("camera_detection", "cam_01", {"class": "car", "confidence": 0.88}, 1002.0),
        ]

        rec = Recorder(recording_path)
        rec.start()
        for et, src, data, ts in events_in:
            rec.record(et, source=src, data=data, timestamp=ts)
        rec.stop()

        player = Player(recording_path)
        events_out = list(player.events())

        assert len(events_out) == len(events_in)
        for (et, src, data, ts), event in zip(events_in, events_out):
            assert event.event_type == et
            assert event.source == src
            assert event.data == data
            assert event.ts == ts

    def test_multiple_sessions_same_file(self, tmp_path: Path):
        """Two sequential recordings to different files, both valid."""
        path1 = tmp_path / "session1.jsonl"
        path2 = tmp_path / "session2.jsonl"

        with Recorder(path1) as r1:
            r1.record("ble_sighting", source="n1", timestamp=100.0)

        with Recorder(path2) as r2:
            r2.record("wifi_probe", source="n2", timestamp=200.0)

        s1 = Session.from_file(path1)
        s2 = Session.from_file(path2)
        assert s1.session_id != s2.session_id
        assert s1.event_count == 1
        assert s2.event_count == 1

    def test_large_batch_recording(self, recording_path: Path):
        """Record many events and verify count."""
        rec = Recorder(recording_path)
        rec.start()
        for i in range(500):
            rec.record("ble_sighting", source=f"node_{i % 10}", data={"idx": i}, timestamp=1000.0 + i * 0.1)
        summary = rec.stop()
        assert summary["event_count"] == 500

        player = Player(recording_path)
        assert player.event_count == 500

    def test_replay_speed_control(self, recording_path: Path):
        """Verify that speed factor affects replay timing."""
        rec = Recorder(recording_path)
        rec.start()
        # Two events 1 second apart
        rec.record("ble_sighting", timestamp=1000.0)
        rec.record("ble_sighting", timestamp=1001.0)
        rec.stop()

        # Replay at 100x speed — should be nearly instant
        player = Player(recording_path, speed=100.0)
        start = time.monotonic()
        events = list(player.replay())
        elapsed = time.monotonic() - start

        assert len(events) == 2
        # At 100x, 1s gap becomes 0.01s — allow generous margin
        assert elapsed < 0.5
