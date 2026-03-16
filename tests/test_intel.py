# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the intelligence, reconnaissance, and fog of war module."""

from __future__ import annotations

import math
import time

import pytest

from tritium_lib.sim_engine.intel import (
    FogOfWar,
    IntelEngine,
    IntelFusion,
    IntelReport,
    IntelType,
    ReconMission,
)


# ===========================================================================
# IntelType enum
# ===========================================================================

class TestIntelType:
    def test_sigint_value(self):
        assert IntelType.SIGINT.value == "sigint"

    def test_humint_value(self):
        assert IntelType.HUMINT.value == "humint"

    def test_imint_value(self):
        assert IntelType.IMINT.value == "imint"

    def test_elint_value(self):
        assert IntelType.ELINT.value == "elint"

    def test_osint_value(self):
        assert IntelType.OSINT.value == "osint"

    def test_all_types_count(self):
        assert len(IntelType) == 5

    def test_from_value(self):
        assert IntelType("sigint") is IntelType.SIGINT


# ===========================================================================
# IntelReport dataclass
# ===========================================================================

class TestIntelReport:
    def _make_report(self, **overrides) -> IntelReport:
        defaults = dict(
            report_id="r1",
            intel_type=IntelType.IMINT,
            source_id="scout-1",
            subject_id="enemy-1",
            position=(100.0, 200.0),
            accuracy=5.0,
            confidence=0.8,
            content="Spotted infantry",
            timestamp=1000.0,
            expires=1300.0,
            alliance="blue",
        )
        defaults.update(overrides)
        return IntelReport(**defaults)

    def test_creation(self):
        r = self._make_report()
        assert r.report_id == "r1"
        assert r.intel_type is IntelType.IMINT
        assert r.confidence == 0.8

    def test_subject_optional(self):
        r = self._make_report(subject_id=None)
        assert r.subject_id is None

    def test_position_tuple(self):
        r = self._make_report()
        assert r.position == (100.0, 200.0)

    def test_expires_after_timestamp(self):
        r = self._make_report()
        assert r.expires > r.timestamp


# ===========================================================================
# ReconMission dataclass
# ===========================================================================

class TestReconMission:
    def test_creation(self):
        m = ReconMission(
            mission_id="m1",
            mission_type="area_recon",
            assigned_unit_id="scout-1",
            waypoints=[(10.0, 10.0), (50.0, 50.0)],
        )
        assert m.status == "planning"
        assert len(m.waypoints) == 2
        assert m.gathered_intel == []

    def test_default_status(self):
        m = ReconMission(
            mission_id="m2",
            mission_type="route_recon",
            assigned_unit_id="scout-2",
            waypoints=[],
        )
        assert m.status == "planning"
        assert m.end_time is None


# ===========================================================================
# FogOfWar
# ===========================================================================

class TestFogOfWar:
    def test_init_dimensions(self):
        fog = FogOfWar(grid_size=(50, 50), cell_size=10.0)
        assert fog.grid_width == 50
        assert fog.grid_height == 50
        assert fog.cell_size == 10.0

    def test_empty_visibility(self):
        fog = FogOfWar((20, 20), 5.0)
        assert not fog.is_visible("blue", (10.0, 10.0))
        assert not fog.is_explored("blue", (10.0, 10.0))

    def test_update_visibility_reveals_cells(self):
        fog = FogOfWar((20, 20), 5.0)
        fog.update_visibility("blue", [((50.0, 50.0), 20.0)])
        assert fog.is_visible("blue", (50.0, 50.0))
        assert fog.is_explored("blue", (50.0, 50.0))

    def test_visibility_range_limit(self):
        fog = FogOfWar((100, 100), 5.0)
        fog.update_visibility("blue", [((50.0, 50.0), 10.0)])
        # Center should be visible
        assert fog.is_visible("blue", (50.0, 50.0))
        # Far away should not be visible
        assert not fog.is_visible("blue", (200.0, 200.0))

    def test_explored_persists_after_moving(self):
        fog = FogOfWar((100, 100), 5.0)
        # First: observe at (50, 50)
        fog.update_visibility("blue", [((50.0, 50.0), 15.0)])
        assert fog.is_visible("blue", (50.0, 50.0))

        # Move observer to (200, 200)
        fog.update_visibility("blue", [((200.0, 200.0), 15.0)])
        # Old position is no longer visible but is explored
        assert not fog.is_visible("blue", (50.0, 50.0))
        assert fog.is_explored("blue", (50.0, 50.0))
        # New position is visible
        assert fog.is_visible("blue", (200.0, 200.0))

    def test_multiple_observers(self):
        fog = FogOfWar((100, 100), 5.0)
        fog.update_visibility("blue", [
            ((50.0, 50.0), 15.0),
            ((100.0, 100.0), 15.0),
        ])
        assert fog.is_visible("blue", (50.0, 50.0))
        assert fog.is_visible("blue", (100.0, 100.0))

    def test_separate_alliances(self):
        fog = FogOfWar((100, 100), 5.0)
        fog.update_visibility("blue", [((50.0, 50.0), 15.0)])
        fog.update_visibility("red", [((200.0, 200.0), 15.0)])
        assert fog.is_visible("blue", (50.0, 50.0))
        assert not fog.is_visible("blue", (200.0, 200.0))
        assert fog.is_visible("red", (200.0, 200.0))
        assert not fog.is_visible("red", (50.0, 50.0))

    def test_get_visible_entities(self):
        fog = FogOfWar((100, 100), 5.0)
        fog.update_visibility("blue", [((50.0, 50.0), 20.0)])
        entities = {
            "e1": (52.0, 52.0),    # close, should be visible
            "e2": (400.0, 400.0),  # far, not visible
        }
        visible = fog.get_visible_entities("blue", entities)
        assert "e1" in visible
        assert "e2" not in visible

    def test_get_visible_entities_empty(self):
        fog = FogOfWar((20, 20), 5.0)
        visible = fog.get_visible_entities("blue", {"e1": (10.0, 10.0)})
        assert len(visible) == 0

    def test_to_three_js_structure(self):
        fog = FogOfWar((10, 10), 5.0)
        fog.update_visibility("blue", [((25.0, 25.0), 10.0)])
        data = fog.to_three_js("blue")
        assert "fog_grid" in data
        fg = data["fog_grid"]
        assert fg["width"] == 10
        assert fg["height"] == 10
        assert fg["cell_size"] == 5.0
        assert fg["fog_color"] == "#0a0a0f"
        assert fg["explored_alpha"] == 0.5
        assert fg["hidden_alpha"] == 0.9
        assert isinstance(fg["visible"], list)
        assert isinstance(fg["explored"], list)

    def test_to_three_js_explored_excludes_visible(self):
        fog = FogOfWar((20, 20), 5.0)
        fog.update_visibility("blue", [((25.0, 25.0), 10.0)])
        # Move observer away
        fog.update_visibility("blue", [((75.0, 75.0), 10.0)])
        data = fog.to_three_js("blue")
        fg = data["fog_grid"]
        # explored should not contain currently visible cells
        visible_set = set(map(tuple, fg["visible"]))
        explored_set = set(map(tuple, fg["explored"]))
        assert visible_set.isdisjoint(explored_set)

    def test_pos_to_cell_clamping(self):
        fog = FogOfWar((10, 10), 5.0)
        # Negative position clamps to (0, 0)
        cell = fog._pos_to_cell((-10.0, -10.0))
        assert cell == (0, 0)
        # Beyond grid clamps to max
        cell = fog._pos_to_cell((999.0, 999.0))
        assert cell == (9, 9)

    def test_cells_in_radius_not_empty(self):
        fog = FogOfWar((20, 20), 5.0)
        cells = fog._cells_in_radius((50.0, 50.0), 10.0)
        assert len(cells) > 0

    def test_cells_in_radius_circular(self):
        """Cells should form a roughly circular pattern, not a square."""
        fog = FogOfWar((100, 100), 1.0)
        cells = fog._cells_in_radius((50.0, 50.0), 10.0)
        # A circle of radius 10 has area ~314. Square would be 441.
        # Allow some discretization error.
        assert len(cells) < 400
        assert len(cells) > 200


# ===========================================================================
# IntelFusion
# ===========================================================================

class TestIntelFusion:
    def _make_report(self, **overrides) -> IntelReport:
        defaults = dict(
            report_id="r1",
            intel_type=IntelType.IMINT,
            source_id="scout",
            subject_id="target-1",
            position=(100.0, 100.0),
            accuracy=10.0,
            confidence=0.6,
            content="Observed",
            timestamp=1000.0,
            expires=1300.0,
            alliance="blue",
        )
        defaults.update(overrides)
        return IntelReport(**defaults)

    def test_fuse_empty_raises(self):
        with pytest.raises(ValueError):
            IntelFusion.fuse_reports([])

    def test_fuse_single_returns_same(self):
        r = self._make_report()
        fused = IntelFusion.fuse_reports([r])
        assert fused is r

    def test_fuse_two_reports_position(self):
        r1 = self._make_report(position=(100.0, 100.0), confidence=0.5)
        r2 = self._make_report(report_id="r2", position=(110.0, 110.0), confidence=0.5)
        fused = IntelFusion.fuse_reports([r1, r2])
        # Equal confidence -> position should be midpoint
        assert abs(fused.position[0] - 105.0) < 1.0
        assert abs(fused.position[1] - 105.0) < 1.0

    def test_fuse_weighted_position(self):
        r1 = self._make_report(position=(100.0, 100.0), confidence=0.9)
        r2 = self._make_report(report_id="r2", position=(200.0, 200.0), confidence=0.1)
        fused = IntelFusion.fuse_reports([r1, r2])
        # High-confidence report should dominate
        assert fused.position[0] < 150.0
        assert fused.position[1] < 150.0

    def test_fuse_improves_accuracy(self):
        r1 = self._make_report(accuracy=20.0)
        r2 = self._make_report(report_id="r2", accuracy=20.0)
        fused = IntelFusion.fuse_reports([r1, r2])
        # 1/sqrt(2) scaling -> ~14.1
        assert fused.accuracy < 20.0
        assert fused.accuracy > 10.0

    def test_fuse_boosts_confidence(self):
        r1 = self._make_report(confidence=0.6)
        r2 = self._make_report(report_id="r2", confidence=0.6)
        fused = IntelFusion.fuse_reports([r1, r2])
        assert fused.confidence > 0.6

    def test_fuse_confidence_caps_at_099(self):
        reports = [
            self._make_report(report_id=f"r{i}", confidence=0.95)
            for i in range(10)
        ]
        fused = IntelFusion.fuse_reports(reports)
        assert fused.confidence <= 0.99

    def test_fuse_takes_latest_timestamp(self):
        r1 = self._make_report(timestamp=1000.0)
        r2 = self._make_report(report_id="r2", timestamp=2000.0)
        fused = IntelFusion.fuse_reports([r1, r2])
        assert fused.timestamp == 2000.0

    def test_fuse_takes_latest_expires(self):
        r1 = self._make_report(expires=1300.0)
        r2 = self._make_report(report_id="r2", expires=2300.0)
        fused = IntelFusion.fuse_reports([r1, r2])
        assert fused.expires == 2300.0

    def test_fuse_report_id_starts_with_fused(self):
        r1 = self._make_report()
        r2 = self._make_report(report_id="r2")
        fused = IntelFusion.fuse_reports([r1, r2])
        assert fused.report_id.startswith("fused-")

    def test_fuse_dominant_type(self):
        r1 = self._make_report(intel_type=IntelType.SIGINT)
        r2 = self._make_report(report_id="r2", intel_type=IntelType.IMINT)
        r3 = self._make_report(report_id="r3", intel_type=IntelType.IMINT)
        fused = IntelFusion.fuse_reports([r1, r2, r3])
        assert fused.intel_type is IntelType.IMINT

    def test_fuse_content_mentions_count(self):
        r1 = self._make_report(content="A")
        r2 = self._make_report(report_id="r2", content="B")
        fused = IntelFusion.fuse_reports([r1, r2])
        assert "2 reports" in fused.content


# ===========================================================================
# IntelEngine — SIGINT
# ===========================================================================

class TestIntelEngineSIGINT:
    def _make_engine(self) -> IntelEngine:
        e = IntelEngine(grid_size=(20, 20), cell_size=5.0)
        e._current_time = 1000.0
        return e

    def test_sigint_intercept_in_range(self):
        eng = self._make_engine()
        msgs = [{"position": (50.0, 50.0), "content": "Attack north", "sender_id": "r1", "encrypted": False}]
        reports = eng.gather_sigint((45.0, 45.0), msgs, "blue", listen_range=100.0)
        assert len(reports) == 1
        assert "Attack north" in reports[0].content

    def test_sigint_out_of_range(self):
        eng = self._make_engine()
        msgs = [{"position": (500.0, 500.0), "content": "Hello", "sender_id": "r1", "encrypted": False}]
        reports = eng.gather_sigint((0.0, 0.0), msgs, "blue", listen_range=100.0)
        assert len(reports) == 0

    def test_sigint_encrypted_message(self):
        eng = self._make_engine()
        msgs = [{"position": (10.0, 10.0), "content": "secret", "sender_id": "r1", "encrypted": True}]
        reports = eng.gather_sigint((5.0, 5.0), msgs, "blue", listen_range=100.0)
        assert len(reports) == 1
        assert "Encrypted" in reports[0].content
        assert "secret" not in reports[0].content

    def test_sigint_confidence_decreases_with_distance(self):
        eng = self._make_engine()
        close_msgs = [{"position": (1.0, 0.0), "content": "hi", "sender_id": "r1", "encrypted": False}]
        far_msgs = [{"position": (90.0, 0.0), "content": "hi", "sender_id": "r2", "encrypted": False}]
        close = eng.gather_sigint((0.0, 0.0), close_msgs, "blue", listen_range=100.0)
        far = eng.gather_sigint((0.0, 0.0), far_msgs, "blue", listen_range=100.0)
        assert close[0].confidence > far[0].confidence

    def test_sigint_reports_stored(self):
        eng = self._make_engine()
        msgs = [{"position": (10.0, 10.0), "content": "msg", "sender_id": "r1", "encrypted": False}]
        eng.gather_sigint((5.0, 5.0), msgs, "blue")
        assert len(eng.reports) == 1


# ===========================================================================
# IntelEngine — IMINT
# ===========================================================================

class TestIntelEngineIMINT:
    def _make_engine(self) -> IntelEngine:
        e = IntelEngine(grid_size=(20, 20), cell_size=5.0)
        e._current_time = 1000.0
        return e

    def test_imint_within_range(self):
        eng = self._make_engine()
        entities = {"e1": (55.0, 55.0)}
        reports = eng.gather_imint((50.0, 50.0), 20.0, entities, "blue")
        assert len(reports) == 1
        assert reports[0].subject_id == "e1"

    def test_imint_out_of_range(self):
        eng = self._make_engine()
        entities = {"e1": (500.0, 500.0)}
        reports = eng.gather_imint((50.0, 50.0), 20.0, entities, "blue")
        assert len(reports) == 0

    def test_imint_multiple_entities(self):
        eng = self._make_engine()
        entities = {"e1": (52.0, 50.0), "e2": (48.0, 50.0), "e3": (999.0, 999.0)}
        reports = eng.gather_imint((50.0, 50.0), 20.0, entities, "blue")
        assert len(reports) == 2
        subjects = {r.subject_id for r in reports}
        assert subjects == {"e1", "e2"}

    def test_imint_closer_better_confidence(self):
        eng = self._make_engine()
        ent_close = {"e1": (51.0, 50.0)}
        ent_far = {"e2": (68.0, 50.0)}
        r_close = eng.gather_imint((50.0, 50.0), 20.0, ent_close, "blue")
        r_far = eng.gather_imint((50.0, 50.0), 20.0, ent_far, "blue")
        assert r_close[0].confidence > r_far[0].confidence

    def test_imint_type_is_imint(self):
        eng = self._make_engine()
        reports = eng.gather_imint((50.0, 50.0), 20.0, {"e1": (52.0, 50.0)}, "blue")
        assert reports[0].intel_type is IntelType.IMINT


# ===========================================================================
# IntelEngine — ELINT
# ===========================================================================

class TestIntelEngineELINT:
    def _make_engine(self) -> IntelEngine:
        e = IntelEngine(grid_size=(20, 20), cell_size=5.0)
        e._current_time = 1000.0
        return e

    def test_elint_in_range(self):
        eng = self._make_engine()
        emitters = [{"position": (100.0, 100.0), "emitter_id": "radar-1", "frequency_mhz": 9400.0, "power_w": 50.0}]
        reports = eng.gather_elint((80.0, 80.0), emitters, "blue", sensor_range=500.0)
        assert len(reports) == 1
        assert "radar-1" in reports[0].content
        assert "9400.0MHz" in reports[0].content

    def test_elint_out_of_range(self):
        eng = self._make_engine()
        emitters = [{"position": (5000.0, 5000.0), "emitter_id": "r1", "frequency_mhz": 100.0}]
        reports = eng.gather_elint((0.0, 0.0), emitters, "blue", sensor_range=100.0)
        assert len(reports) == 0

    def test_elint_type(self):
        eng = self._make_engine()
        emitters = [{"position": (10.0, 10.0), "emitter_id": "r1", "frequency_mhz": 100.0}]
        reports = eng.gather_elint((5.0, 5.0), emitters, "blue", sensor_range=100.0)
        assert reports[0].intel_type is IntelType.ELINT

    def test_elint_reports_stored(self):
        eng = self._make_engine()
        emitters = [{"position": (10.0, 10.0), "emitter_id": "r1", "frequency_mhz": 100.0}]
        eng.gather_elint((5.0, 5.0), emitters, "blue", sensor_range=100.0)
        assert len(eng.reports) == 1


# ===========================================================================
# IntelEngine — Recon Missions
# ===========================================================================

class TestReconMissions:
    def _make_engine(self) -> IntelEngine:
        e = IntelEngine(grid_size=(100, 100), cell_size=5.0)
        e._current_time = 1000.0
        return e

    def test_create_mission(self):
        eng = self._make_engine()
        m = eng.create_recon_mission("area_recon", "scout-1", [(10.0, 10.0), (50.0, 50.0)])
        assert m.mission_type == "area_recon"
        assert m.assigned_unit_id == "scout-1"
        assert m.status == "planning"
        assert m.mission_id in eng.missions

    def test_mission_activates_on_tick(self):
        eng = self._make_engine()
        m = eng.create_recon_mission("route_recon", "scout-1", [(10.0, 10.0)])
        m.start_time = 1000.0
        eng.tick(1.0, {"blue": [((0.0, 0.0), 10.0)]}, {})
        assert m.status == "active" or m.status == "complete"

    def test_mission_completes_after_waypoints(self):
        eng = self._make_engine()
        m = eng.create_recon_mission("point_recon", "scout-1", [(10.0, 10.0)])
        m.start_time = 999.0  # already past
        eng.tick(1.0, {}, {})
        # After one tick with one waypoint, should complete
        eng.tick(1.0, {}, {})
        assert m.status == "complete"
        assert m.end_time is not None

    def test_mission_gathers_intel(self):
        eng = self._make_engine()
        wps = [(50.0, 50.0)]
        m = eng.create_recon_mission("point_recon", "scout-1", wps)
        m.start_time = 999.0
        entities = {"enemy-1": (55.0, 55.0)}
        eng.tick(1.0, {}, entities)
        # Mission should have gathered intel about nearby entity
        assert len(m.gathered_intel) >= 1

    def test_aborted_mission(self):
        eng = self._make_engine()
        m = eng.create_recon_mission("surveillance", "scout-1", [(10.0, 10.0)])
        m.status = "aborted"
        eng.tick(1.0, {}, {})
        assert m.status == "aborted"


# ===========================================================================
# IntelEngine — tick and expiry
# ===========================================================================

class TestIntelEngineTick:
    def _make_engine(self) -> IntelEngine:
        e = IntelEngine(grid_size=(20, 20), cell_size=5.0)
        e._current_time = 1000.0
        return e

    def test_tick_advances_time(self):
        eng = self._make_engine()
        eng.tick(5.0, {}, {})
        assert eng._current_time == 1005.0

    def test_tick_expires_stale_intel(self):
        eng = self._make_engine()
        eng.reports.append(IntelReport(
            report_id="old",
            intel_type=IntelType.IMINT,
            source_id="s1",
            subject_id="e1",
            position=(10.0, 10.0),
            accuracy=5.0,
            confidence=0.5,
            content="old report",
            timestamp=500.0,
            expires=900.0,  # already expired at t=1000
            alliance="blue",
        ))
        eng.tick(1.0, {}, {})
        # Report should be expired and removed
        old_reports = [r for r in eng.reports if r.report_id == "old"]
        assert len(old_reports) == 0

    def test_tick_keeps_valid_intel(self):
        eng = self._make_engine()
        eng.reports.append(IntelReport(
            report_id="fresh",
            intel_type=IntelType.IMINT,
            source_id="s1",
            subject_id="e1",
            position=(10.0, 10.0),
            accuracy=5.0,
            confidence=0.5,
            content="fresh report",
            timestamp=999.0,
            expires=2000.0,
            alliance="blue",
        ))
        eng.tick(1.0, {}, {})
        fresh = [r for r in eng.reports if r.report_id == "fresh"]
        assert len(fresh) == 1

    def test_tick_updates_fog(self):
        eng = self._make_engine()
        eng.tick(1.0, {"blue": [((50.0, 50.0), 20.0)]}, {})
        assert eng.fog.is_visible("blue", (50.0, 50.0))

    def test_tick_merges_conflicting_reports(self):
        eng = self._make_engine()
        # Add two reports about the same subject
        for i in range(3):
            eng.reports.append(IntelReport(
                report_id=f"r{i}",
                intel_type=IntelType.IMINT,
                source_id=f"s{i}",
                subject_id="target-1",
                position=(100.0 + i * 5, 100.0),
                accuracy=10.0,
                confidence=0.5,
                content=f"Report {i}",
                timestamp=1000.0 + i,
                expires=2000.0,
                alliance="blue",
            ))
        initial_count = len(eng.reports)
        eng.tick(1.0, {}, {})
        # Should have fewer reports after fusion
        assert len(eng.reports) < initial_count


# ===========================================================================
# IntelEngine — intel picture
# ===========================================================================

class TestIntelPicture:
    def _make_engine(self) -> IntelEngine:
        e = IntelEngine(grid_size=(20, 20), cell_size=5.0)
        e._current_time = 1000.0
        return e

    def test_get_intel_picture_empty(self):
        eng = self._make_engine()
        pic = eng.get_intel_picture("blue")
        assert pic["alliance"] == "blue"
        assert pic["reports"] == []
        assert pic["threat_count"] == 0

    def test_get_intel_picture_with_reports(self):
        eng = self._make_engine()
        eng.gather_imint((50.0, 50.0), 20.0, {"e1": (55.0, 55.0)}, "blue")
        pic = eng.get_intel_picture("blue")
        assert len(pic["reports"]) == 1
        assert pic["threat_count"] == 1

    def test_get_intel_picture_only_own_alliance(self):
        eng = self._make_engine()
        eng.gather_imint((50.0, 50.0), 20.0, {"e1": (55.0, 55.0)}, "blue")
        eng.gather_imint((50.0, 50.0), 20.0, {"e2": (55.0, 55.0)}, "red")
        pic_blue = eng.get_intel_picture("blue")
        pic_red = eng.get_intel_picture("red")
        assert len(pic_blue["reports"]) == 1
        assert len(pic_red["reports"]) == 1

    def test_coverage_pct(self):
        eng = self._make_engine()
        eng.tick(1.0, {"blue": [((50.0, 50.0), 20.0)]}, {})
        pic = eng.get_intel_picture("blue")
        assert pic["coverage_pct"] > 0.0
        assert pic["coverage_pct"] <= 100.0

    def test_intel_picture_report_structure(self):
        eng = self._make_engine()
        eng.gather_imint((50.0, 50.0), 20.0, {"e1": (55.0, 55.0)}, "blue")
        pic = eng.get_intel_picture("blue")
        r = pic["reports"][0]
        assert "report_id" in r
        assert "type" in r
        assert "position" in r
        assert "confidence" in r
        assert "content" in r
        assert isinstance(r["position"], list)


# ===========================================================================
# IntelEngine — threat estimate
# ===========================================================================

class TestThreatEstimate:
    def _make_engine(self) -> IntelEngine:
        e = IntelEngine(grid_size=(100, 100), cell_size=5.0)
        e._current_time = 1000.0
        return e

    def test_threat_estimate_empty(self):
        eng = self._make_engine()
        est = eng.get_threat_estimate("blue", ((50.0, 50.0), 100.0))
        assert est["estimated_contacts"] == 0
        assert est["confidence"] == 0.0
        assert est["assessed_threat"] == "none"

    def test_threat_estimate_with_contacts(self):
        eng = self._make_engine()
        eng.gather_imint((50.0, 50.0), 100.0, {"e1": (55.0, 55.0), "e2": (60.0, 60.0)}, "blue")
        est = eng.get_threat_estimate("blue", ((55.0, 55.0), 50.0))
        assert est["estimated_contacts"] >= 1
        assert est["confidence"] > 0.0
        assert est["assessed_threat"] in ("low", "moderate", "high", "critical")

    def test_threat_estimate_outside_area(self):
        eng = self._make_engine()
        eng.gather_imint((50.0, 50.0), 20.0, {"e1": (55.0, 55.0)}, "blue")
        est = eng.get_threat_estimate("blue", ((400.0, 400.0), 10.0))
        assert est["estimated_contacts"] == 0

    def test_threat_levels(self):
        eng = self._make_engine()
        # Add many contacts in same area
        entities = {f"e{i}": (50.0 + i, 50.0) for i in range(12)}
        eng.gather_imint((50.0, 50.0), 100.0, entities, "blue")
        est = eng.get_threat_estimate("blue", ((55.0, 50.0), 100.0))
        assert est["assessed_threat"] == "critical"

    def test_threat_estimate_structure(self):
        eng = self._make_engine()
        est = eng.get_threat_estimate("blue", ((0.0, 0.0), 50.0))
        assert "area_center" in est
        assert "area_radius" in est
        assert "estimated_contacts" in est
        assert "confidence" in est
        assert "reports" in est
        assert "assessed_threat" in est


# ===========================================================================
# IntelEngine — Three.js export
# ===========================================================================

class TestIntelEngineThreeJS:
    def _make_engine(self) -> IntelEngine:
        e = IntelEngine(grid_size=(20, 20), cell_size=5.0)
        e._current_time = 1000.0
        return e

    def test_to_three_js_structure(self):
        eng = self._make_engine()
        data = eng.to_three_js("blue")
        assert "fog_grid" in data
        assert "intel_markers" in data
        assert "recon_paths" in data

    def test_to_three_js_with_reports(self):
        eng = self._make_engine()
        eng.gather_imint((50.0, 50.0), 20.0, {"e1": (55.0, 55.0)}, "blue")
        data = eng.to_three_js("blue")
        assert len(data["intel_markers"]) >= 1
        m = data["intel_markers"][0]
        assert "id" in m
        assert "type" in m
        assert "position" in m

    def test_to_three_js_with_recon(self):
        eng = self._make_engine()
        eng.create_recon_mission("area_recon", "scout-1", [(10.0, 10.0), (50.0, 50.0)])
        data = eng.to_three_js("blue")
        assert len(data["recon_paths"]) == 1
        rp = data["recon_paths"][0]
        assert rp["type"] == "area_recon"
        assert len(rp["waypoints"]) == 2

    def test_to_three_js_fog_included(self):
        eng = self._make_engine()
        eng.tick(1.0, {"blue": [((50.0, 50.0), 20.0)]}, {})
        data = eng.to_three_js("blue")
        assert len(data["fog_grid"]["visible"]) > 0


# ===========================================================================
# Integration / Edge cases
# ===========================================================================

class TestIntelIntegration:
    def test_full_cycle(self):
        """Full intel cycle: gather, tick, picture, threat."""
        eng = IntelEngine(grid_size=(100, 100), cell_size=5.0)
        eng._current_time = 1000.0

        entities = {"tank-1": (200.0, 200.0), "infantry-1": (210.0, 205.0)}

        # Gather IMINT
        eng.gather_imint((190.0, 190.0), 50.0, entities, "blue")
        # Gather SIGINT
        eng.gather_sigint(
            (180.0, 180.0),
            [{"position": (200.0, 200.0), "content": "Moving east", "sender_id": "tank-1", "encrypted": False}],
            "blue",
        )
        # Tick with observers
        eng.tick(1.0, {"blue": [((190.0, 190.0), 50.0)]}, entities)

        # Check picture
        pic = eng.get_intel_picture("blue")
        assert pic["threat_count"] >= 1

        # Check threat
        est = eng.get_threat_estimate("blue", ((200.0, 200.0), 50.0))
        assert est["estimated_contacts"] >= 1

    def test_multi_alliance_isolation(self):
        """Each alliance sees only its own intel."""
        eng = IntelEngine(grid_size=(50, 50), cell_size=5.0)
        eng._current_time = 1000.0
        eng.gather_imint((50.0, 50.0), 30.0, {"e1": (55.0, 55.0)}, "blue")
        eng.gather_imint((150.0, 150.0), 30.0, {"e2": (155.0, 155.0)}, "red")

        blue_pic = eng.get_intel_picture("blue")
        red_pic = eng.get_intel_picture("red")

        blue_subjects = {r["subject_id"] for r in blue_pic["reports"]}
        red_subjects = {r["subject_id"] for r in red_pic["reports"]}

        assert "e1" in blue_subjects
        assert "e2" not in blue_subjects
        assert "e2" in red_subjects
        assert "e1" not in red_subjects

    def test_stale_intel_removal_over_time(self):
        """Intel should expire after enough ticks."""
        eng = IntelEngine(grid_size=(20, 20), cell_size=5.0)
        eng._current_time = 1000.0
        eng.gather_imint((50.0, 50.0), 20.0, {"e1": (55.0, 55.0)}, "blue")
        assert len(eng.reports) == 1

        # Advance far past expiry (IMINT expires at +120s)
        for _ in range(130):
            eng.tick(1.0, {}, {})

        pic = eng.get_intel_picture("blue")
        assert len(pic["reports"]) == 0

    def test_engine_default_grid(self):
        eng = IntelEngine()
        assert eng.fog.grid_width == 100
        assert eng.fog.grid_height == 100

    def test_gather_sigint_custom_time(self):
        eng = IntelEngine(grid_size=(20, 20), cell_size=5.0)
        eng._current_time = 1000.0
        msgs = [{"position": (10.0, 10.0), "content": "hi", "sender_id": "r1", "encrypted": False}]
        reports = eng.gather_sigint((5.0, 5.0), msgs, "blue", intercept_time=5000.0)
        assert reports[0].timestamp == 5000.0

    def test_gather_imint_custom_time(self):
        eng = IntelEngine(grid_size=(20, 20), cell_size=5.0)
        reports = eng.gather_imint((50.0, 50.0), 20.0, {"e1": (55.0, 55.0)}, "blue", observation_time=9999.0)
        assert reports[0].timestamp == 9999.0

    def test_gather_elint_custom_time(self):
        eng = IntelEngine(grid_size=(20, 20), cell_size=5.0)
        emitters = [{"position": (10.0, 10.0), "emitter_id": "r1", "frequency_mhz": 100.0}]
        reports = eng.gather_elint((5.0, 5.0), emitters, "blue", detection_time=7777.0)
        assert reports[0].timestamp == 7777.0

    def test_mission_id_unique(self):
        eng = IntelEngine(grid_size=(20, 20), cell_size=5.0)
        m1 = eng.create_recon_mission("area_recon", "s1", [(0.0, 0.0)])
        m2 = eng.create_recon_mission("route_recon", "s2", [(0.0, 0.0)])
        assert m1.mission_id != m2.mission_id
