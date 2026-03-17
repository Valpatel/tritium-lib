# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine.replay — recording, playback, analysis, export."""

from __future__ import annotations

import gzip
import json
import os
import tempfile

import pytest

from tritium_lib.sim_engine.replay import (
    ReplayFrame,
    ReplayRecorder,
    ReplayPlayer,
    ReplayAnalyzer,
    ReplayExporter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unit(uid: str, x: float = 0.0, y: float = 0.0, alliance: str = "friendly",
               health: float = 100.0, status: str = "idle", name: str = "",
               damage_dealt: float = 0.0, damage_taken: float = 0.0) -> dict:
    return {
        "id": uid,
        "name": name or uid,
        "unit_type": "infantry",
        "alliance": alliance,
        "x": x,
        "y": y,
        "heading": 0.0,
        "health": health,
        "max_health": 100.0,
        "is_alive": health > 0,
        "status": status,
        "morale": 1.0,
        "suppression": 0.0,
        "weapon": "m4a1",
        "squad_id": None,
        "kill_count": 0,
        "damage_dealt": damage_dealt,
        "damage_taken": damage_taken,
    }


def _make_frame(tick: int, time: float, units: list[dict] | None = None,
                events: list[dict] | None = None,
                render_data: dict | None = None) -> ReplayFrame:
    return ReplayFrame(
        tick=tick,
        time=time,
        units=units or [],
        events=events or [],
        render_data=render_data or {},
    )


def _make_kill_event(target_id: str, source_id: str = "") -> dict:
    return {"type": "unit_killed", "target_id": target_id, "source_id": source_id}


def _make_fire_event(unit_id: str, weapon: str = "m4a1") -> dict:
    return {"type": "fire", "unit_id": unit_id, "weapon": weapon, "target": (100, 100)}


def _sample_frames(n: int = 10) -> list[ReplayFrame]:
    """Generate n sample frames with one unit moving diagonally."""
    frames = []
    for i in range(n):
        events = []
        if i == 5:
            events.append(_make_kill_event("enemy_1", "u_1"))
        frames.append(_make_frame(
            tick=i,
            time=i * 0.1,
            units=[
                _make_unit("u_1", x=float(i), y=float(i), alliance="friendly"),
                _make_unit("enemy_1", x=50.0, y=50.0, alliance="hostile",
                           health=100.0 if i < 5 else 0.0,
                           status="idle" if i < 5 else "dead"),
            ],
            events=events,
            render_data={"tick": i},
        ))
    return frames


# ===========================================================================
# ReplayFrame tests
# ===========================================================================


class TestReplayFrame:
    def test_create(self):
        f = _make_frame(0, 0.0)
        assert f.tick == 0
        assert f.time == 0.0
        assert f.units == []
        assert f.events == []
        assert f.render_data == {}

    def test_create_with_data(self):
        units = [_make_unit("u1")]
        events = [{"type": "spawn"}]
        f = _make_frame(5, 1.5, units=units, events=events, render_data={"tick": 5})
        assert f.tick == 5
        assert f.time == 1.5
        assert len(f.units) == 1
        assert len(f.events) == 1
        assert f.render_data["tick"] == 5

    def test_to_dict(self):
        f = _make_frame(3, 0.3, units=[_make_unit("u1")], events=[{"type": "test"}])
        d = f.to_dict()
        assert d["tick"] == 3
        assert d["time"] == 0.3
        assert len(d["units"]) == 1
        assert len(d["events"]) == 1

    def test_from_dict(self):
        d = {"tick": 7, "time": 0.7, "units": [_make_unit("u1")],
             "events": [{"type": "x"}], "render_data": {"a": 1}}
        f = ReplayFrame.from_dict(d)
        assert f.tick == 7
        assert f.time == 0.7
        assert len(f.units) == 1
        assert f.render_data == {"a": 1}

    def test_from_dict_missing_optional(self):
        d = {"tick": 0, "time": 0.0}
        f = ReplayFrame.from_dict(d)
        assert f.units == []
        assert f.events == []
        assert f.render_data == {}

    def test_roundtrip(self):
        original = _make_frame(10, 1.0, units=[_make_unit("u1")],
                               events=[{"type": "fire"}], render_data={"k": "v"})
        d = original.to_dict()
        restored = ReplayFrame.from_dict(d)
        assert restored.tick == original.tick
        assert restored.time == original.time
        assert restored.units == original.units
        assert restored.events == original.events
        assert restored.render_data == original.render_data


# ===========================================================================
# ReplayRecorder tests
# ===========================================================================


class TestReplayRecorder:
    def test_create_default(self):
        r = ReplayRecorder()
        assert r.frames == []
        assert r.max_frames == 36000
        assert "start_time" in r.metadata

    def test_create_with_metadata(self):
        r = ReplayRecorder(metadata={"preset": "urban_combat"})
        assert r.metadata["preset"] == "urban_combat"
        assert "start_time" in r.metadata

    def test_record_frame(self):
        r = ReplayRecorder()
        r.record_frame(0, 0.0, [_make_unit("u1")], [], {})
        assert len(r.frames) == 1
        assert r.frames[0].tick == 0

    def test_record_multiple_frames(self):
        r = ReplayRecorder()
        for i in range(10):
            r.record_frame(i, i * 0.1, [], [], {})
        assert len(r.frames) == 10
        assert r.frames[9].tick == 9

    def test_max_frames_enforced(self):
        r = ReplayRecorder(max_frames=5)
        for i in range(10):
            r.record_frame(i, i * 0.1, [], [], {})
        assert len(r.frames) == 5
        assert r.frames[0].tick == 5  # oldest frames dropped

    def test_save_and_load(self):
        r = ReplayRecorder(metadata={"preset": "test"})
        for i in range(5):
            r.record_frame(i, i * 0.1, [_make_unit("u1", x=float(i))], [], {"tick": i})
        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as tmp:
            path = tmp.name
        try:
            r.save(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

            # Verify it is valid gzip + JSON
            with gzip.open(path, "rb") as fh:
                data = json.loads(fh.read().decode("utf-8"))
            assert data["version"] == 1
            assert data["frame_count"] == 5
            assert len(data["frames"]) == 5
            assert data["metadata"]["preset"] == "test"
        finally:
            os.unlink(path)

    def test_get_size_mb(self):
        r = ReplayRecorder()
        r.record_frame(0, 0.0, [_make_unit("u1")], [{"type": "spawn"}], {"tick": 0})
        size = r.get_size_mb()
        assert size > 0.0
        assert size < 1.0  # single frame should be tiny

    def test_get_size_mb_empty(self):
        r = ReplayRecorder()
        assert r.get_size_mb() == 0.0 or r.get_size_mb() < 0.001

    def test_record_from_world(self):
        """Test record_from_world with a real World instance."""
        from tritium_lib.sim_engine.world import World, WorldConfig
        config = WorldConfig(map_size=(100, 100), seed=42, enable_weather=False,
                             enable_destruction=False, enable_los=False)
        world = World(config)
        world.spawn_unit("infantry", "Alpha", "friendly", (10.0, 10.0))
        world.spawn_unit("infantry", "Bravo", "hostile", (90.0, 90.0))
        world.tick()

        recorder = ReplayRecorder()
        recorder.record_from_world(world, world.events)
        assert len(recorder.frames) == 1
        frame = recorder.frames[0]
        assert len(frame.units) == 2
        assert frame.tick == world.tick_count
        # Units should have position data
        u1 = frame.units[0]
        assert "x" in u1
        assert "y" in u1
        assert "alliance" in u1

    def test_save_creates_parent_dirs(self):
        r = ReplayRecorder()
        r.record_frame(0, 0.0, [], [], {})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "replay.json.gz")
            r.save(path)
            assert os.path.exists(path)

    def test_metadata_preserved(self):
        meta = {"preset": "urban", "player_names": ["Alice", "Bob"],
                "factions": ["friendly", "hostile"]}
        r = ReplayRecorder(metadata=meta)
        r.record_frame(0, 0.0, [], [], {})
        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as tmp:
            path = tmp.name
        try:
            r.save(path)
            with gzip.open(path, "rb") as fh:
                data = json.loads(fh.read().decode("utf-8"))
            assert data["metadata"]["preset"] == "urban"
            assert data["metadata"]["player_names"] == ["Alice", "Bob"]
        finally:
            os.unlink(path)


# ===========================================================================
# ReplayPlayer tests
# ===========================================================================


class TestReplayPlayer:
    def test_create(self):
        p = ReplayPlayer()
        assert p.frames == []
        assert p.current_frame == 0
        assert p.playback_speed == 1.0
        assert not p.is_playing

    def test_load_from_recorder(self):
        r = ReplayRecorder()
        for i in range(5):
            r.record_frame(i, i * 0.1, [], [], {})
        p = ReplayPlayer()
        p.load_from_recorder(r)
        assert len(p.frames) == 5

    def test_load_from_file(self):
        r = ReplayRecorder(metadata={"preset": "test"})
        for i in range(3):
            r.record_frame(i, i * 0.5, [_make_unit("u1")], [], {})
        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as tmp:
            path = tmp.name
        try:
            r.save(path)
            p = ReplayPlayer()
            p.load(path)
            assert len(p.frames) == 3
            assert p.metadata["preset"] == "test"
            assert p.current_frame == 0
            assert not p.is_playing
        finally:
            os.unlink(path)

    def test_play_pause_stop(self):
        p = ReplayPlayer()
        p.frames = [_make_frame(0, 0.0)]
        p.play()
        assert p.is_playing
        p.pause()
        assert not p.is_playing
        p.play()
        p.stop()
        assert not p.is_playing
        assert p.current_frame == 0

    def test_next_frame(self):
        p = ReplayPlayer()
        p.frames = _sample_frames(5)
        p.play()
        f = p.next_frame()
        assert f is not None
        assert f.tick == 0
        assert p.current_frame == 1

    def test_next_frame_exhausted(self):
        p = ReplayPlayer()
        p.frames = _sample_frames(2)
        p.play()
        assert p.next_frame() is not None  # frame 0
        assert p.next_frame() is not None  # frame 1
        assert p.next_frame() is None      # exhausted
        assert not p.is_playing

    def test_next_frame_not_playing(self):
        p = ReplayPlayer()
        p.frames = _sample_frames(3)
        # not playing
        assert p.next_frame() is None

    def test_prev_frame(self):
        p = ReplayPlayer()
        p.frames = _sample_frames(5)
        p.current_frame = 3
        f = p.prev_frame()
        assert f is not None
        assert f.tick == 2
        assert p.current_frame == 2

    def test_prev_frame_at_start(self):
        p = ReplayPlayer()
        p.frames = _sample_frames(5)
        p.current_frame = 0
        assert p.prev_frame() is None

    def test_prev_frame_empty(self):
        p = ReplayPlayer()
        assert p.prev_frame() is None

    def test_seek_by_tick(self):
        p = ReplayPlayer()
        p.frames = _sample_frames(10)
        p.seek(5)
        assert p.current_frame == 5

    def test_seek_closest_tick(self):
        p = ReplayPlayer()
        # Frames with ticks 0, 2, 4, 6, 8
        p.frames = [_make_frame(i * 2, i * 0.2) for i in range(5)]
        p.seek(3)  # closest to tick 2 (idx 1) or tick 4 (idx 2)
        assert p.current_frame in (1, 2)  # either is acceptable

    def test_seek_empty(self):
        p = ReplayPlayer()
        p.seek(5)  # should not crash

    def test_seek_time(self):
        p = ReplayPlayer()
        p.frames = _sample_frames(10)
        p.seek_time(0.5)
        assert p.current_frame == 5

    def test_seek_time_empty(self):
        p = ReplayPlayer()
        p.seek_time(1.0)  # should not crash

    def test_get_frame_by_tick(self):
        p = ReplayPlayer()
        p.frames = _sample_frames(5)
        f = p.get_frame(3)
        assert f is not None
        assert f.tick == 3

    def test_get_frame_missing_tick(self):
        p = ReplayPlayer()
        p.frames = _sample_frames(5)
        assert p.get_frame(999) is None

    def test_set_speed(self):
        p = ReplayPlayer()
        p.set_speed(2.0)
        assert p.playback_speed == 2.0
        p.set_speed(0.25)
        assert p.playback_speed == 0.25

    def test_set_speed_clamped(self):
        p = ReplayPlayer()
        p.set_speed(0.01)
        assert p.playback_speed == 0.1
        p.set_speed(100.0)
        assert p.playback_speed == 16.0

    def test_total_duration(self):
        p = ReplayPlayer()
        p.frames = _sample_frames(10)
        assert p.total_duration() == pytest.approx(0.9, abs=0.01)

    def test_total_duration_empty(self):
        p = ReplayPlayer()
        assert p.total_duration() == 0.0

    def test_total_frames(self):
        p = ReplayPlayer()
        p.frames = _sample_frames(7)
        assert p.total_frames() == 7

    def test_progress_start(self):
        p = ReplayPlayer()
        p.frames = _sample_frames(10)
        p.current_frame = 0
        assert p.progress() == 0.0

    def test_progress_middle(self):
        p = ReplayPlayer()
        p.frames = _sample_frames(10)
        p.current_frame = 5
        assert p.progress() == pytest.approx(0.5)

    def test_progress_end(self):
        p = ReplayPlayer()
        p.frames = _sample_frames(10)
        p.current_frame = 10
        assert p.progress() == pytest.approx(1.0)

    def test_progress_empty(self):
        p = ReplayPlayer()
        assert p.progress() == 0.0

    def test_full_playback_loop(self):
        """Play all frames and verify they arrive in order."""
        p = ReplayPlayer()
        p.frames = _sample_frames(10)
        p.play()
        ticks = []
        while True:
            f = p.next_frame()
            if f is None:
                break
            ticks.append(f.tick)
        assert ticks == list(range(10))
        assert not p.is_playing


# ===========================================================================
# ReplayAnalyzer tests
# ===========================================================================


class TestReplayAnalyzer:
    def test_kill_timeline(self):
        frames = _sample_frames(10)
        a = ReplayAnalyzer(frames)
        kills = a.kill_timeline()
        assert len(kills) == 1
        assert kills[0]["tick"] == 5
        assert kills[0]["target_id"] == "enemy_1"
        assert kills[0]["source_id"] == "u_1"

    def test_kill_timeline_empty(self):
        frames = [_make_frame(i, i * 0.1) for i in range(5)]
        a = ReplayAnalyzer(frames)
        assert a.kill_timeline() == []

    def test_kill_timeline_multiple(self):
        frames = [
            _make_frame(0, 0.0, events=[_make_kill_event("a", "x")]),
            _make_frame(1, 0.1),
            _make_frame(2, 0.2, events=[_make_kill_event("b", "y"),
                                        _make_kill_event("c", "z")]),
        ]
        a = ReplayAnalyzer(frames)
        kills = a.kill_timeline()
        assert len(kills) == 3
        assert kills[0]["tick"] == 0
        assert kills[2]["tick"] == 2

    def test_movement_heatmap(self):
        frames = _sample_frames(10)
        a = ReplayAnalyzer(frames)
        hm = a.movement_heatmap("u_1", grid_size=5.0)
        assert hm["unit_id"] == "u_1"
        assert hm["grid_size"] == 5.0
        assert hm["total_samples"] == 10
        assert len(hm["cells"]) > 0

    def test_movement_heatmap_unknown_unit(self):
        frames = _sample_frames(5)
        a = ReplayAnalyzer(frames)
        hm = a.movement_heatmap("nonexistent")
        assert hm["total_samples"] == 0
        assert hm["cells"] == []

    def test_movement_heatmap_stationary(self):
        """Unit that doesn't move should have one cell."""
        frames = [
            _make_frame(i, i * 0.1, units=[_make_unit("u1", x=5.0, y=5.0)])
            for i in range(10)
        ]
        a = ReplayAnalyzer(frames)
        hm = a.movement_heatmap("u1", grid_size=10.0)
        assert len(hm["cells"]) == 1
        assert hm["cells"][0]["count"] == 10

    def test_damage_graph(self):
        frames = [
            _make_frame(0, 0.0, units=[
                _make_unit("u1", damage_dealt=0.0, damage_taken=0.0),
            ]),
            _make_frame(1, 0.1, units=[
                _make_unit("u1", damage_dealt=10.0, damage_taken=5.0),
            ]),
        ]
        a = ReplayAnalyzer(frames)
        dg = a.damage_graph()
        assert "u1" in dg["units"]
        assert dg["units"]["u1"]["damage_dealt"] == [0.0, 10.0]
        assert dg["units"]["u1"]["damage_taken"] == [0.0, 5.0]

    def test_damage_graph_empty(self):
        a = ReplayAnalyzer([])
        dg = a.damage_graph()
        assert dg["units"] == {}

    def test_decisive_moment(self):
        frames = _sample_frames(10)
        a = ReplayAnalyzer(frames)
        dm = a.decisive_moment()
        assert dm is not None
        assert dm.tick == 5  # the frame with the kill event

    def test_decisive_moment_empty(self):
        a = ReplayAnalyzer([])
        assert a.decisive_moment() is None

    def test_decisive_moment_tie(self):
        """When multiple frames have the same max events, return one of them."""
        frames = [
            _make_frame(0, 0.0, events=[{"type": "a"}]),
            _make_frame(1, 0.1, events=[{"type": "b"}]),
        ]
        a = ReplayAnalyzer(frames)
        dm = a.decisive_moment()
        assert dm is not None
        assert dm.tick in (0, 1)

    def test_unit_path(self):
        frames = _sample_frames(10)
        a = ReplayAnalyzer(frames)
        path = a.unit_path("u_1")
        assert len(path) == 10
        assert path[0] == (0.0, 0.0)
        assert path[9] == (9.0, 9.0)

    def test_unit_path_unknown(self):
        frames = _sample_frames(5)
        a = ReplayAnalyzer(frames)
        assert a.unit_path("nonexistent") == []

    def test_unit_path_stationary(self):
        frames = [
            _make_frame(i, i * 0.1, units=[_make_unit("u1", x=5.0, y=5.0)])
            for i in range(3)
        ]
        a = ReplayAnalyzer(frames)
        path = a.unit_path("u1")
        assert path == [(5.0, 5.0)] * 3

    def test_summary(self):
        frames = _sample_frames(10)
        a = ReplayAnalyzer(frames)
        s = a.summary()
        assert s["total_frames"] == 10
        assert s["duration"] == pytest.approx(0.9, abs=0.01)
        assert s["total_kills"] == 1
        assert s["unique_units"] == 2
        assert "friendly" in s["factions"]
        assert "hostile" in s["factions"]
        assert s["first_tick"] == 0
        assert s["last_tick"] == 9

    def test_summary_empty(self):
        a = ReplayAnalyzer([])
        s = a.summary()
        assert s["total_frames"] == 0
        assert s["duration"] == 0.0
        assert s["total_kills"] == 0
        assert s["unique_units"] == 0

    def test_summary_no_events(self):
        frames = [_make_frame(i, i * 0.1, units=[_make_unit("u1")]) for i in range(5)]
        a = ReplayAnalyzer(frames)
        s = a.summary()
        assert s["total_events"] == 0
        assert s["total_kills"] == 0


# ===========================================================================
# ReplayExporter tests
# ===========================================================================


class TestReplayExporter:
    def test_to_csv(self):
        frames = _sample_frames(3)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
            path = tmp.name
        try:
            ReplayExporter.to_csv(frames, path)
            with open(path) as fh:
                lines = fh.readlines()
            header = lines[0].strip()
            assert "tick" in header
            assert "unit_id" in header
            # 3 frames * 2 units = 6 data lines + 1 header
            assert len(lines) == 7
        finally:
            os.unlink(path)

    def test_to_csv_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            path = tmp.name
        try:
            ReplayExporter.to_csv([], path)
            with open(path) as fh:
                lines = fh.readlines()
            assert len(lines) == 1  # header only
        finally:
            os.unlink(path)

    def test_to_csv_creates_dirs(self):
        frames = [_make_frame(0, 0.0, units=[_make_unit("u1")])]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "replay.csv")
            ReplayExporter.to_csv(frames, path)
            assert os.path.exists(path)

    def test_to_geojson(self):
        frames = _sample_frames(10)
        gj = ReplayExporter.to_geojson(frames)
        assert gj["type"] == "FeatureCollection"
        # enemy_1 is stationary so it gets filtered (needs >= 2 distinct points)
        assert len(gj["features"]) >= 1
        for feat in gj["features"]:
            assert feat["type"] == "Feature"
            assert feat["geometry"]["type"] == "LineString"
            assert "unit_id" in feat["properties"]

    def test_to_geojson_empty(self):
        gj = ReplayExporter.to_geojson([])
        assert gj["type"] == "FeatureCollection"
        assert gj["features"] == []

    def test_to_geojson_stationary_unit(self):
        """Stationary unit should be excluded (needs >= 2 distinct points)."""
        frames = [
            _make_frame(i, i * 0.1, units=[_make_unit("u1", x=5.0, y=5.0)])
            for i in range(5)
        ]
        gj = ReplayExporter.to_geojson(frames)
        # All points are the same, so after dedup there's only 1 point -> excluded
        assert len(gj["features"]) == 0

    def test_to_geojson_moving_unit(self):
        frames = [
            _make_frame(i, i * 0.1, units=[_make_unit("u1", x=float(i), y=float(i))])
            for i in range(5)
        ]
        gj = ReplayExporter.to_geojson(frames)
        assert len(gj["features"]) == 1
        coords = gj["features"][0]["geometry"]["coordinates"]
        assert len(coords) == 5


# ===========================================================================
# Integration tests
# ===========================================================================


class TestIntegration:
    def test_record_save_load_play(self):
        """Full round-trip: record -> save -> load -> play."""
        rec = ReplayRecorder(metadata={"preset": "integration_test"})
        for i in range(20):
            events = []
            if i == 10:
                events.append(_make_kill_event("enemy", "hero"))
            rec.record_frame(
                tick=i, time=i * 0.05,
                units=[_make_unit("hero", x=float(i), y=float(i * 2))],
                events=events,
                render_data={"tick": i},
            )

        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as tmp:
            path = tmp.name
        try:
            rec.save(path)

            player = ReplayPlayer()
            player.load(path)
            assert player.total_frames() == 20
            assert player.metadata["preset"] == "integration_test"

            player.play()
            all_frames = []
            while True:
                f = player.next_frame()
                if f is None:
                    break
                all_frames.append(f)
            assert len(all_frames) == 20

            analyzer = ReplayAnalyzer(all_frames)
            kills = analyzer.kill_timeline()
            assert len(kills) == 1
            assert kills[0]["tick"] == 10

            summary = analyzer.summary()
            assert summary["total_frames"] == 20
            assert summary["total_kills"] == 1
        finally:
            os.unlink(path)

    def test_record_from_world_full_cycle(self):
        """Record a World simulation, analyze results."""
        from tritium_lib.sim_engine.world import World, WorldConfig

        config = WorldConfig(
            map_size=(100, 100), seed=42,
            enable_weather=False, enable_destruction=False, enable_los=False,
        )
        world = World(config)
        world.spawn_unit("infantry", "Alpha", "friendly", (10.0, 50.0))
        world.spawn_unit("infantry", "Bravo", "hostile", (90.0, 50.0))

        rec = ReplayRecorder(metadata={"preset": "world_test"})
        for _ in range(50):
            world.tick()
            rec.record_from_world(world, world.events)

        assert len(rec.frames) == 50

        analyzer = ReplayAnalyzer(rec.frames)
        summary = analyzer.summary()
        assert summary["total_frames"] == 50
        assert summary["unique_units"] == 2

        # Check paths exist for both units
        for frame in rec.frames:
            for u in frame.units:
                uid = u["id"]
                path = analyzer.unit_path(uid)
                assert len(path) == 50

    def test_seek_then_play(self):
        """Seek to a point, then play from there."""
        p = ReplayPlayer()
        p.frames = _sample_frames(20)
        p.seek(10)
        assert p.current_frame == 10
        p.play()
        f = p.next_frame()
        assert f is not None
        assert f.tick == 10

    def test_backward_navigation(self):
        """Navigate backward through frames."""
        p = ReplayPlayer()
        p.frames = _sample_frames(10)
        p.current_frame = 5
        ticks = []
        while True:
            f = p.prev_frame()
            if f is None:
                break
            ticks.append(f.tick)
        assert ticks == [4, 3, 2, 1, 0]

    def test_analyze_then_export(self):
        """Analyze and export to both CSV and GeoJSON."""
        frames = _sample_frames(10)
        analyzer = ReplayAnalyzer(frames)

        # CSV export
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            csv_path = tmp.name
        try:
            ReplayExporter.to_csv(frames, csv_path)
            with open(csv_path) as fh:
                lines = fh.readlines()
            assert len(lines) > 1

            # GeoJSON export
            gj = ReplayExporter.to_geojson(frames)
            assert len(gj["features"]) >= 1

            # Summary should match
            summary = analyzer.summary()
            assert summary["total_frames"] == 10
        finally:
            os.unlink(csv_path)
