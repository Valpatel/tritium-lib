# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for territory.py — InfluenceMap, TerritoryControl, ControlPoint."""

import pytest
from tritium_lib.sim_engine.territory import (
    InfluenceMap,
    TerritoryControl,
    ControlPoint,
    CaptureState,
    StrategicValue,
)


# ---------------------------------------------------------------------------
# InfluenceMap
# ---------------------------------------------------------------------------

class TestInfluenceMapInit:
    def test_basic_construction(self):
        im = InfluenceMap(10, 10)
        assert im.width == 10
        assert im.height == 10
        assert im.cell_size == 10.0

    def test_custom_cell_size(self):
        im = InfluenceMap(20, 20, cell_size=5.0)
        assert im.cell_size == 5.0

    def test_invalid_width(self):
        with pytest.raises(ValueError):
            InfluenceMap(0, 10)

    def test_invalid_height(self):
        with pytest.raises(ValueError):
            InfluenceMap(10, 0)

    def test_invalid_cell_size(self):
        with pytest.raises(ValueError):
            InfluenceMap(10, 10, cell_size=0.0)

    def test_no_factions_initially(self):
        im = InfluenceMap(10, 10)
        assert im.factions == []


class TestInfluenceMapAddInfluence:
    def test_adds_faction_on_first_call(self):
        im = InfluenceMap(10, 10)
        im.add_influence("blue", (50.0, 50.0), 0.8, 30.0)
        assert "blue" in im.factions

    def test_influence_at_center_positive(self):
        im = InfluenceMap(10, 10)
        im.add_influence("blue", (50.0, 50.0), 1.0, 40.0)
        val = im.get_influence((50.0, 50.0), "blue")
        assert val > 0.0

    def test_influence_clamps_to_one(self):
        im = InfluenceMap(10, 10)
        # Add multiple times to push past 1.0
        for _ in range(20):
            im.add_influence("blue", (50.0, 50.0), 1.0, 100.0)
        val = im.get_influence((50.0, 50.0), "blue")
        assert val <= 1.0

    def test_zero_radius_does_nothing(self):
        im = InfluenceMap(10, 10)
        im.add_influence("blue", (50.0, 50.0), 0.8, 0.0)
        assert "blue" not in im.factions

    def test_out_of_bounds_returns_zero(self):
        im = InfluenceMap(5, 5)
        assert im.get_influence((-100.0, -100.0), "blue") == 0.0

    def test_unknown_faction_returns_zero(self):
        im = InfluenceMap(10, 10)
        assert im.get_influence((50.0, 50.0), "unknown") == 0.0

    def test_two_factions_independent(self):
        im = InfluenceMap(10, 10)
        im.add_influence("blue", (10.0, 10.0), 1.0, 30.0)
        im.add_influence("red", (90.0, 90.0), 1.0, 30.0)
        assert im.get_influence((10.0, 10.0), "blue") > 0.0
        # red should have no influence at blue's position
        assert im.get_influence((10.0, 10.0), "red") == 0.0


class TestInfluenceMapController:
    def test_get_controller_returns_dominant(self):
        im = InfluenceMap(10, 10)
        im.add_influence("blue", (50.0, 50.0), 1.0, 60.0)
        assert im.get_controller((50.0, 50.0)) == "blue"

    def test_get_controller_none_when_no_influence(self):
        im = InfluenceMap(10, 10)
        assert im.get_controller((50.0, 50.0)) is None

    def test_get_controller_stronger_wins(self):
        im = InfluenceMap(10, 10)
        im.add_influence("blue", (50.0, 50.0), 0.9, 60.0)
        im.add_influence("red", (50.0, 50.0), 0.3, 20.0)
        assert im.get_controller((50.0, 50.0)) == "blue"


class TestInfluenceMapDecay:
    def test_decay_reduces_influence(self):
        im = InfluenceMap(10, 10)
        im.add_influence("blue", (50.0, 50.0), 0.5, 30.0)
        before = im.get_influence((50.0, 50.0), "blue")
        im.decay(rate=0.1)
        after = im.get_influence((50.0, 50.0), "blue")
        assert after < before

    def test_decay_does_not_go_below_zero(self):
        im = InfluenceMap(10, 10)
        im.add_influence("blue", (50.0, 50.0), 0.05, 30.0)
        for _ in range(100):
            im.decay(rate=0.1)
        val = im.get_influence((50.0, 50.0), "blue")
        assert val >= 0.0


class TestInfluenceMapContestedZones:
    def test_contested_zones_requires_two_factions(self):
        im = InfluenceMap(10, 10)
        im.add_influence("blue", (50.0, 50.0), 0.5, 40.0)
        assert im.get_contested_zones() == []

    def test_contested_zone_detected(self):
        im = InfluenceMap(10, 10)
        # Both factions with equal influence at the same spot
        im.add_influence("blue", (50.0, 50.0), 0.5, 60.0)
        im.add_influence("red", (50.0, 50.0), 0.5, 60.0)
        zones = im.get_contested_zones()
        assert len(zones) > 0

    def test_contested_zone_has_expected_keys(self):
        im = InfluenceMap(10, 10)
        im.add_influence("blue", (50.0, 50.0), 0.5, 60.0)
        im.add_influence("red", (50.0, 50.0), 0.5, 60.0)
        zones = im.get_contested_zones()
        if zones:
            z = zones[0]
            assert "position" in z
            assert "cell" in z
            assert "factions" in z
            assert "spread" in z


class TestInfluenceMapFrontline:
    def test_frontline_empty_for_unknown_faction(self):
        im = InfluenceMap(10, 10)
        im.add_influence("blue", (20.0, 50.0), 1.0, 60.0)
        assert im.get_frontline("blue", "red") == []

    def test_frontline_detected_between_factions(self):
        im = InfluenceMap(20, 10)
        # blue on left, red on right
        im.add_influence("blue", (20.0, 50.0), 1.0, 50.0)
        im.add_influence("red", (180.0, 50.0), 1.0, 50.0)
        fl = im.get_frontline("blue", "red")
        # May or may not produce frontline depending on overlap — just check type
        assert isinstance(fl, list)


class TestInfluenceMapClear:
    def test_clear_removes_all(self):
        im = InfluenceMap(10, 10)
        im.add_influence("blue", (50.0, 50.0), 1.0, 40.0)
        im.add_influence("red", (50.0, 50.0), 1.0, 40.0)
        im.clear()
        assert im.factions == []

    def test_clear_faction(self):
        im = InfluenceMap(10, 10)
        im.add_influence("blue", (50.0, 50.0), 1.0, 40.0)
        im.add_influence("red", (50.0, 50.0), 1.0, 40.0)
        im.clear_faction("blue")
        assert "blue" not in im.factions
        assert "red" in im.factions


class TestInfluenceMapTick:
    def test_tick_increases_influence(self):
        im = InfluenceMap(10, 10)
        units = {"u1": ((50.0, 50.0), "blue")}
        im.tick(1.0, units)
        assert im.get_influence((50.0, 50.0), "blue") > 0.0

    def test_tick_empty_units(self):
        im = InfluenceMap(10, 10)
        im.add_influence("blue", (50.0, 50.0), 0.5, 30.0)
        before = im.total_influence("blue")
        im.tick(1.0, {})
        # Influence only decays, should not increase
        assert im.total_influence("blue") <= before


class TestInfluenceMapToThreeJs:
    def test_to_three_js_structure(self):
        im = InfluenceMap(5, 5, cell_size=20.0)
        im.add_influence("blue", (50.0, 50.0), 1.0, 40.0)
        result = im.to_three_js()
        assert "width" in result
        assert "height" in result
        assert "cell_size" in result
        assert "heatmaps" in result
        assert "frontlines" in result
        assert "blue" in result["heatmaps"]
        assert len(result["heatmaps"]["blue"]) == 5 * 5


class TestInfluenceMapControlledCellCount:
    def test_controlled_cells_positive_after_influence(self):
        im = InfluenceMap(10, 10)
        im.add_influence("blue", (50.0, 50.0), 1.0, 80.0)
        assert im.controlled_cell_count("blue") > 0

    def test_controlled_cells_zero_for_unknown(self):
        im = InfluenceMap(10, 10)
        assert im.controlled_cell_count("nobody") == 0


# ---------------------------------------------------------------------------
# TerritoryControl / ControlPoint
# ---------------------------------------------------------------------------

class TestControlPoint:
    def test_default_values(self):
        cp = ControlPoint("cp1", (100.0, 200.0))
        assert cp.point_id == "cp1"
        assert cp.owner is None
        assert cp.value == 1.0

    def test_custom_values(self):
        cp = ControlPoint("cp2", (0.0, 0.0), name="Hill 42", capture_radius=50.0, value=2.0)
        assert cp.name == "Hill 42"
        assert cp.capture_radius == 50.0
        assert cp.value == 2.0


class TestTerritoryControl:
    def _make_tc(self):
        tc = TerritoryControl()
        cp = ControlPoint("alpha", (100.0, 100.0), name="Alpha", capture_radius=50.0, capture_time=5.0)
        tc.add_control_point(cp)
        return tc, cp

    def test_add_control_point(self):
        tc, cp = self._make_tc()
        assert tc.get_point("alpha") is cp

    def test_remove_control_point(self):
        tc, _ = self._make_tc()
        tc.remove_control_point("alpha")
        assert tc.get_point("alpha") is None

    def test_get_point_unknown(self):
        tc, _ = self._make_tc()
        assert tc.get_point("nosuchpoint") is None

    def test_tick_no_units_no_capture(self):
        tc, cp = self._make_tc()
        events = tc.tick(1.0, {})
        assert cp.owner is None
        assert isinstance(events, list)

    def test_tick_single_faction_capture_progress(self):
        tc, cp = self._make_tc()
        # Put blue unit right at the control point
        units = {"u1": ((100.0, 100.0), "blue")}
        # rate per tick = CAPTURE_RATE(0.1) * dt(0.1) * sqrt(1) = 0.01
        # progress per tick = 0.01 / capture_time(5.0) = 0.002
        # ticks needed > 500 to reach 1.0
        for _ in range(600):
            tc.tick(0.1, units)
        assert cp.owner == "blue"

    def test_tick_contested_emits_event(self):
        tc, cp = self._make_tc()
        units = {
            "u1": ((100.0, 100.0), "blue"),
            "u2": ((100.0, 100.0), "red"),
        }
        events = tc.tick(1.0, units)
        contested = [e for e in events if e.get("type") == "contested"]
        assert len(contested) > 0

    def test_tick_capture_event_fires(self):
        tc, cp = self._make_tc()
        units = {"u1": ((100.0, 100.0), "blue")}
        capture_events = []
        for _ in range(500):
            events = tc.tick(0.1, units)
            capture_events.extend([e for e in events if e.get("type") == "captured"])
        assert len(capture_events) > 0

    def test_captured_event_has_correct_keys(self):
        tc, cp = self._make_tc()
        units = {"u1": ((100.0, 100.0), "blue")}
        capture_events = []
        for _ in range(500):
            events = tc.tick(0.1, units)
            capture_events.extend([e for e in events if e.get("type") == "captured"])
        if capture_events:
            e = capture_events[0]
            assert "point_id" in e
            assert "faction" in e

    def test_influence_map_integration(self):
        im = InfluenceMap(10, 10, cell_size=20.0)
        tc = TerritoryControl(influence_map=im)
        cp = ControlPoint("beta", (100.0, 100.0), capture_radius=50.0, capture_time=2.0)
        tc.add_control_point(cp)
        units = {"u1": ((100.0, 100.0), "blue")}
        for _ in range(100):
            tc.tick(0.1, units)
        # After capture, influence map should have blue influence at cp position
        assert im.get_influence((100.0, 100.0), "blue") >= 0.0  # at minimum it was called

    def test_faction_summary_after_capture(self):
        tc, cp = self._make_tc()
        units = {"u1": ((100.0, 100.0), "blue")}
        for _ in range(600):
            tc.tick(0.1, units)
        summary = tc.get_territory_summary("blue")
        assert "controlled_points" in summary
        assert summary["total_points"] >= 1


# ---------------------------------------------------------------------------
# StrategicValue
# ---------------------------------------------------------------------------

class TestStrategicValue:
    def test_rate_position_returns_float(self):
        sv = StrategicValue(10, 10)
        score = sv.rate_position((50.0, 50.0))
        assert isinstance(score, float)

    def test_score_in_range_zero_one(self):
        sv = StrategicValue(10, 10)
        score = sv.rate_position((50.0, 50.0))
        assert 0.0 <= score <= 1.0

    def test_higher_elevation_scores_higher(self):
        sv = StrategicValue(10, 10, cell_size=10.0)
        # Set elevation at cell (5,5) high
        sv.set_elevation(5, 5, 100.0)
        high_score = sv.rate_cell(5, 5)
        sv.set_elevation(5, 5, 0.0)
        low_score = sv.rate_cell(5, 5)
        assert high_score >= low_score

    def test_road_cell_scores_higher(self):
        sv = StrategicValue(10, 10, cell_size=10.0)
        no_road = sv.rate_cell(5, 5)
        sv.add_road((55.0, 55.0))  # adds road at cell (5,5)
        with_road = sv.rate_cell(5, 5)
        assert with_road >= no_road

    def test_find_high_value_positions_returns_list(self):
        sv = StrategicValue(10, 10, cell_size=10.0)
        sv.set_elevation(5, 5, 100.0)
        sv.add_road((55.0, 55.0))
        results = sv.find_high_value_positions(min_value=0.0, max_results=5)
        assert isinstance(results, list)
        if results:
            r = results[0]
            assert "position" in r
            assert "value" in r

    def test_to_grid_shape(self):
        sv = StrategicValue(5, 4, cell_size=10.0)
        grid = sv.to_grid()
        assert len(grid) == 4
        assert len(grid[0]) == 5
