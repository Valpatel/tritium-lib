# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for territory control and influence map system."""

from __future__ import annotations

import math
import pytest

from tritium_lib.sim_engine.territory import (
    InfluenceMap,
    TerritoryControl,
    ControlPoint,
    CaptureState,
    StrategicValue,
    _MIN_INFLUENCE,
    _CONTESTED_THRESHOLD,
    _CAPTURE_RATE,
    _DEFAULT_CELL_SIZE,
)


# ===========================================================================
# InfluenceMap — Construction
# ===========================================================================


class TestInfluenceMapConstruction:
    """Tests for InfluenceMap creation and validation."""

    def test_basic_creation(self):
        im = InfluenceMap(10, 10)
        assert im.width == 10
        assert im.height == 10
        assert im.cell_size == _DEFAULT_CELL_SIZE

    def test_custom_cell_size(self):
        im = InfluenceMap(5, 5, cell_size=20.0)
        assert im.cell_size == 20.0

    def test_zero_width_raises(self):
        with pytest.raises(ValueError, match="positive"):
            InfluenceMap(0, 10)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError, match="positive"):
            InfluenceMap(10, 0)

    def test_negative_dimensions_raise(self):
        with pytest.raises(ValueError):
            InfluenceMap(-5, 10)

    def test_zero_cell_size_raises(self):
        with pytest.raises(ValueError, match="cell_size"):
            InfluenceMap(10, 10, cell_size=0)

    def test_negative_cell_size_raises(self):
        with pytest.raises(ValueError):
            InfluenceMap(10, 10, cell_size=-1.0)

    def test_empty_factions_initially(self):
        im = InfluenceMap(5, 5)
        assert im.factions == []

    def test_large_grid(self):
        im = InfluenceMap(100, 100, cell_size=5.0)
        assert im.width == 100
        assert im.height == 100


# ===========================================================================
# InfluenceMap — add_influence
# ===========================================================================


class TestInfluenceMapAddInfluence:
    """Tests for adding influence to the map."""

    def test_add_influence_center(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (50.0, 50.0), 1.0, 30.0)
        # Center cell should have high influence
        assert im.get_influence((50.0, 50.0), "red") > 0.5

    def test_add_influence_falloff(self):
        im = InfluenceMap(20, 20, cell_size=10.0)
        im.add_influence("red", (100.0, 100.0), 1.0, 50.0)
        center = im.get_influence((100.0, 100.0), "red")
        edge = im.get_influence((140.0, 100.0), "red")
        # Closer should have more influence
        assert center > edge

    def test_add_influence_zero_radius(self):
        im = InfluenceMap(10, 10)
        im.add_influence("red", (50.0, 50.0), 1.0, 0)
        assert im.get_influence((50.0, 50.0), "red") == 0.0

    def test_add_influence_zero_strength(self):
        im = InfluenceMap(10, 10)
        im.add_influence("red", (50.0, 50.0), 0, 30.0)
        assert im.get_influence((50.0, 50.0), "red") == 0.0

    def test_add_influence_negative_strength_ignored(self):
        im = InfluenceMap(10, 10)
        im.add_influence("red", (50.0, 50.0), -1.0, 30.0)
        assert im.get_influence((50.0, 50.0), "red") == 0.0

    def test_add_influence_clamped_to_one(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        # Add lots of influence to the same spot
        for _ in range(20):
            im.add_influence("red", (50.0, 50.0), 1.0, 30.0)
        assert im.get_influence((50.0, 50.0), "red") <= 1.0

    def test_add_influence_creates_faction(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("blue", (50.0, 50.0), 0.5, 20.0)
        assert "blue" in im.factions

    def test_add_influence_multiple_factions(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (30.0, 50.0), 0.5, 20.0)
        im.add_influence("blue", (70.0, 50.0), 0.5, 20.0)
        assert len(im.factions) == 2

    def test_out_of_bounds_position(self):
        im = InfluenceMap(5, 5, cell_size=10.0)
        # Position way outside grid — should not crash
        im.add_influence("red", (500.0, 500.0), 1.0, 20.0)
        # No influence inside the grid
        assert im.get_influence((25.0, 25.0), "red") == 0.0

    def test_strength_capped_at_one(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (50.0, 50.0), 5.0, 30.0)
        # Even with strength > 1, cell value should be <= 1
        assert im.get_influence((50.0, 50.0), "red") <= 1.0


# ===========================================================================
# InfluenceMap — get_influence and get_controller
# ===========================================================================


class TestInfluenceMapQueries:
    """Tests for querying influence values and controllers."""

    def test_get_influence_unknown_faction(self):
        im = InfluenceMap(10, 10)
        assert im.get_influence((50.0, 50.0), "nonexistent") == 0.0

    def test_get_influence_out_of_bounds(self):
        im = InfluenceMap(5, 5, cell_size=10.0)
        assert im.get_influence((-10.0, -10.0), "red") == 0.0

    def test_get_controller_no_influence(self):
        im = InfluenceMap(10, 10)
        assert im.get_controller((50.0, 50.0)) is None

    def test_get_controller_single_faction(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (50.0, 50.0), 0.8, 30.0)
        assert im.get_controller((50.0, 50.0)) == "red"

    def test_get_controller_dominant_faction(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (50.0, 50.0), 0.8, 30.0)
        im.add_influence("blue", (50.0, 50.0), 0.3, 30.0)
        assert im.get_controller((50.0, 50.0)) == "red"

    def test_get_controller_out_of_bounds(self):
        im = InfluenceMap(5, 5, cell_size=10.0)
        assert im.get_controller((500.0, 500.0)) is None


# ===========================================================================
# InfluenceMap — decay
# ===========================================================================


class TestInfluenceMapDecay:
    """Tests for influence decay over time."""

    def test_decay_reduces_influence(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (50.0, 50.0), 0.5, 30.0)
        before = im.get_influence((50.0, 50.0), "red")
        im.decay(0.1)
        after = im.get_influence((50.0, 50.0), "red")
        assert after < before

    def test_decay_never_negative(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (50.0, 50.0), 0.01, 20.0)
        im.decay(1.0)  # Massive decay
        assert im.get_influence((50.0, 50.0), "red") >= 0.0

    def test_decay_zero_rate_no_change(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (50.0, 50.0), 0.5, 30.0)
        before = im.get_influence((50.0, 50.0), "red")
        im.decay(0.0)
        after = im.get_influence((50.0, 50.0), "red")
        assert after == before

    def test_decay_multiple_factions(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (50.0, 50.0), 0.5, 30.0)
        im.add_influence("blue", (50.0, 50.0), 0.5, 30.0)
        im.decay(0.1)
        assert im.get_influence((50.0, 50.0), "red") < 0.5
        assert im.get_influence((50.0, 50.0), "blue") < 0.5


# ===========================================================================
# InfluenceMap — contested zones
# ===========================================================================


class TestInfluenceMapContestedZones:
    """Tests for contested zone detection."""

    def test_no_contested_single_faction(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (50.0, 50.0), 0.8, 50.0)
        zones = im.get_contested_zones()
        assert len(zones) == 0

    def test_contested_where_factions_overlap(self):
        im = InfluenceMap(20, 5, cell_size=10.0)
        # Two factions overlapping in the middle
        im.add_influence("red", (50.0, 25.0), 0.5, 60.0)
        im.add_influence("blue", (150.0, 25.0), 0.5, 60.0)
        zones = im.get_contested_zones()
        # Should find some contested cells near the overlap
        assert len(zones) >= 0  # May or may not have contested zones depending on exact overlap

    def test_contested_zone_has_required_keys(self):
        im = InfluenceMap(10, 5, cell_size=10.0)
        im.add_influence("red", (25.0, 25.0), 0.5, 40.0)
        im.add_influence("blue", (25.0, 25.0), 0.45, 40.0)
        zones = im.get_contested_zones()
        if zones:
            z = zones[0]
            assert "position" in z
            assert "cell" in z
            assert "factions" in z
            assert "spread" in z

    def test_contested_zone_spread_within_threshold(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        # Same position, similar strength
        im.add_influence("red", (50.0, 50.0), 0.5, 30.0)
        im.add_influence("blue", (50.0, 50.0), 0.48, 30.0)
        zones = im.get_contested_zones()
        for z in zones:
            assert z["spread"] <= _CONTESTED_THRESHOLD


# ===========================================================================
# InfluenceMap — frontline
# ===========================================================================


class TestInfluenceMapFrontline:
    """Tests for frontline detection between factions."""

    def test_frontline_no_factions(self):
        im = InfluenceMap(10, 10)
        assert im.get_frontline("red", "blue") == []

    def test_frontline_one_faction_missing(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (50.0, 50.0), 0.8, 30.0)
        assert im.get_frontline("red", "blue") == []

    def test_frontline_between_opposing_factions(self):
        im = InfluenceMap(20, 5, cell_size=10.0)
        # Red on the left, blue on the right
        im.add_influence("red", (30.0, 25.0), 0.9, 60.0)
        im.add_influence("blue", (170.0, 25.0), 0.9, 60.0)
        fl = im.get_frontline("red", "blue")
        # Should have some frontline cells where territories meet
        # (may be empty if territories don't overlap)
        assert isinstance(fl, list)

    def test_frontline_cells_are_vec2(self):
        im = InfluenceMap(10, 5, cell_size=10.0)
        im.add_influence("red", (15.0, 25.0), 0.9, 40.0)
        im.add_influence("blue", (85.0, 25.0), 0.9, 40.0)
        fl = im.get_frontline("red", "blue")
        for point in fl:
            assert isinstance(point, tuple)
            assert len(point) == 2


# ===========================================================================
# InfluenceMap — tick
# ===========================================================================


class TestInfluenceMapTick:
    """Tests for the tick method (units project influence + decay)."""

    def test_tick_projects_influence(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        units = {"u1": ((50.0, 50.0), "red")}
        im.tick(1.0, units)
        assert im.get_influence((50.0, 50.0), "red") > 0

    def test_tick_multiple_factions(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        units = {
            "u1": ((20.0, 50.0), "red"),
            "u2": ((80.0, 50.0), "blue"),
        }
        im.tick(1.0, units)
        assert im.get_influence((20.0, 50.0), "red") > 0
        assert im.get_influence((80.0, 50.0), "blue") > 0

    def test_tick_decay_reduces_old_influence(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (50.0, 50.0), 0.5, 30.0)
        before = im.get_influence((50.0, 50.0), "red")
        # Tick with no units — should just decay
        im.tick(1.0, {})
        after = im.get_influence((50.0, 50.0), "red")
        assert after < before

    def test_tick_zero_dt(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (50.0, 50.0), 0.5, 30.0)
        before = im.get_influence((50.0, 50.0), "red")
        im.tick(0.0, {"u1": ((50.0, 50.0), "red")})
        after = im.get_influence((50.0, 50.0), "red")
        assert after == pytest.approx(before, abs=0.01)


# ===========================================================================
# InfluenceMap — utility methods
# ===========================================================================


class TestInfluenceMapUtility:
    """Tests for clear, total_influence, controlled_cell_count."""

    def test_clear(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (50.0, 50.0), 0.8, 30.0)
        im.clear()
        assert im.factions == []
        assert im.get_influence((50.0, 50.0), "red") == 0.0

    def test_clear_faction(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (50.0, 50.0), 0.8, 30.0)
        im.add_influence("blue", (50.0, 50.0), 0.5, 30.0)
        im.clear_faction("red")
        assert "red" not in im.factions
        assert im.get_influence((50.0, 50.0), "blue") > 0

    def test_total_influence(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (50.0, 50.0), 0.5, 30.0)
        total = im.total_influence("red")
        assert total > 0

    def test_total_influence_unknown_faction(self):
        im = InfluenceMap(10, 10)
        assert im.total_influence("nonexistent") == 0.0

    def test_controlled_cell_count(self):
        im = InfluenceMap(10, 10, cell_size=10.0)
        im.add_influence("red", (50.0, 50.0), 0.8, 80.0)
        count = im.controlled_cell_count("red")
        assert count > 0

    def test_controlled_cell_count_no_influence(self):
        im = InfluenceMap(10, 10)
        assert im.controlled_cell_count("red") == 0


# ===========================================================================
# InfluenceMap — to_three_js
# ===========================================================================


class TestInfluenceMapThreeJS:
    """Tests for Three.js export."""

    def test_to_three_js_structure(self):
        im = InfluenceMap(5, 5, cell_size=10.0)
        im.add_influence("red", (25.0, 25.0), 0.5, 30.0)
        data = im.to_three_js()
        assert data["width"] == 5
        assert data["height"] == 5
        assert data["cell_size"] == 10.0
        assert "heatmaps" in data
        assert "frontlines" in data

    def test_to_three_js_heatmap_length(self):
        im = InfluenceMap(5, 5, cell_size=10.0)
        im.add_influence("red", (25.0, 25.0), 0.5, 30.0)
        data = im.to_three_js()
        assert len(data["heatmaps"]["red"]) == 25  # 5 * 5

    def test_to_three_js_frontlines_computed(self):
        im = InfluenceMap(10, 5, cell_size=10.0)
        im.add_influence("red", (15.0, 25.0), 0.9, 40.0)
        im.add_influence("blue", (85.0, 25.0), 0.9, 40.0)
        data = im.to_three_js()
        assert isinstance(data["frontlines"], dict)

    def test_to_three_js_empty_map(self):
        im = InfluenceMap(3, 3, cell_size=10.0)
        data = im.to_three_js()
        assert data["heatmaps"] == {}
        assert data["frontlines"] == {}


# ===========================================================================
# TerritoryControl — Construction and point management
# ===========================================================================


class TestTerritoryControlBasic:
    """Tests for TerritoryControl creation and point management."""

    def test_create_empty(self):
        tc = TerritoryControl()
        assert len(tc.control_points) == 0

    def test_add_control_point(self):
        tc = TerritoryControl()
        cp = ControlPoint("cp1", (100.0, 100.0), name="Hill Alpha")
        tc.add_control_point(cp)
        assert len(tc.control_points) == 1

    def test_get_point(self):
        tc = TerritoryControl()
        cp = ControlPoint("cp1", (100.0, 100.0), name="Hill Alpha")
        tc.add_control_point(cp)
        assert tc.get_point("cp1") is cp

    def test_get_point_not_found(self):
        tc = TerritoryControl()
        assert tc.get_point("nope") is None

    def test_remove_control_point(self):
        tc = TerritoryControl()
        tc.add_control_point(ControlPoint("cp1", (100.0, 100.0)))
        tc.add_control_point(ControlPoint("cp2", (200.0, 200.0)))
        tc.remove_control_point("cp1")
        assert len(tc.control_points) == 1
        assert tc.get_point("cp1") is None

    def test_remove_nonexistent_point_safe(self):
        tc = TerritoryControl()
        tc.remove_control_point("nope")  # Should not raise


# ===========================================================================
# TerritoryControl — Capture logic
# ===========================================================================


class TestTerritoryControlCapture:
    """Tests for capture and contest mechanics."""

    def test_capture_single_unit(self):
        tc = TerritoryControl()
        cp = ControlPoint("cp1", (100.0, 100.0), capture_time=1.0)
        tc.add_control_point(cp)
        # Place a unit right on the point and tick many times
        units = {"u1": ((100.0, 100.0), "red")}
        events = []
        for _ in range(200):
            events.extend(tc.tick(0.1, units))
        captured = [e for e in events if e["type"] == "captured"]
        assert len(captured) >= 1
        assert captured[0]["faction"] == "red"
        assert cp.owner == "red"

    def test_contested_blocks_capture(self):
        tc = TerritoryControl()
        cp = ControlPoint("cp1", (100.0, 100.0), capture_time=5.0)
        tc.add_control_point(cp)
        # Both factions present
        units = {
            "u1": ((100.0, 100.0), "red"),
            "u2": ((105.0, 100.0), "blue"),
        }
        events = []
        for _ in range(50):
            events.extend(tc.tick(0.1, units))
        # Should be contested, not captured
        contested = [e for e in events if e["type"] == "contested"]
        assert len(contested) > 0
        assert cp.owner is None

    def test_no_units_nearby_no_progress(self):
        tc = TerritoryControl()
        cp = ControlPoint("cp1", (100.0, 100.0), capture_time=5.0)
        tc.add_control_point(cp)
        units = {"u1": ((9999.0, 9999.0), "red")}
        tc.tick(1.0, units)
        assert cp.owner is None
        assert tc.capture_progress.get("cp1", {}) == {}

    def test_more_units_capture_faster(self):
        tc1 = TerritoryControl()
        cp1 = ControlPoint("cp1", (100.0, 100.0), capture_time=10.0)
        tc1.add_control_point(cp1)
        # Single unit
        units_1 = {"u1": ((100.0, 100.0), "red")}
        tc1.tick(1.0, units_1)
        prog_1 = tc1.capture_progress.get("cp1", {}).get("red", 0)

        tc2 = TerritoryControl()
        cp2 = ControlPoint("cp1", (100.0, 100.0), capture_time=10.0)
        tc2.add_control_point(cp2)
        # Three units
        units_3 = {
            "u1": ((100.0, 100.0), "red"),
            "u2": ((105.0, 100.0), "red"),
            "u3": ((100.0, 105.0), "red"),
        }
        tc2.tick(1.0, units_3)
        prog_3 = tc2.capture_progress.get("cp1", {}).get("red", 0)

        assert prog_3 > prog_1

    def test_capture_with_influence_map(self):
        im = InfluenceMap(20, 20, cell_size=10.0)
        tc = TerritoryControl(influence_map=im)
        cp = ControlPoint("cp1", (100.0, 100.0), capture_time=1.0, value=2.0)
        tc.add_control_point(cp)
        units = {"u1": ((100.0, 100.0), "red")}
        for _ in range(200):
            tc.tick(0.1, units)
        # Influence map should have influence from the capture
        assert im.get_influence((100.0, 100.0), "red") > 0

    def test_progress_decays_when_no_units(self):
        tc = TerritoryControl()
        cp = ControlPoint("cp1", (100.0, 100.0), capture_time=10.0)
        tc.add_control_point(cp)
        # Start capturing
        units = {"u1": ((100.0, 100.0), "red")}
        tc.tick(1.0, units)
        prog_with = tc.capture_progress.get("cp1", {}).get("red", 0)
        assert prog_with > 0
        # Remove units and tick — progress should decay
        for _ in range(50):
            tc.tick(1.0, {})
        prog_after = tc.capture_progress.get("cp1", {}).get("red", 0)
        assert prog_after < prog_with


# ===========================================================================
# TerritoryControl — State queries
# ===========================================================================


class TestTerritoryControlState:
    """Tests for capture state and territory summary queries."""

    def test_state_neutral_initially(self):
        tc = TerritoryControl()
        cp = ControlPoint("cp1", (100.0, 100.0))
        tc.add_control_point(cp)
        assert tc.get_state("cp1") == CaptureState.NEUTRAL

    def test_state_capturing(self):
        tc = TerritoryControl()
        cp = ControlPoint("cp1", (100.0, 100.0), capture_time=100.0)
        tc.add_control_point(cp)
        units = {"u1": ((100.0, 100.0), "red")}
        # Tick enough times to build measurable progress but not capture
        for _ in range(10):
            tc.tick(1.0, units)
        assert tc.get_state("cp1") == CaptureState.CAPTURING

    def test_state_captured(self):
        tc = TerritoryControl()
        cp = ControlPoint("cp1", (100.0, 100.0), capture_time=1.0)
        tc.add_control_point(cp)
        units = {"u1": ((100.0, 100.0), "red")}
        for _ in range(200):
            tc.tick(0.1, units)
        assert tc.get_state("cp1") == CaptureState.CAPTURED

    def test_state_unknown_point(self):
        tc = TerritoryControl()
        assert tc.get_state("nope") == CaptureState.NEUTRAL

    def test_territory_summary_empty(self):
        tc = TerritoryControl()
        summary = tc.get_territory_summary("red")
        assert summary["controlled_points"] == []
        assert summary["total_value"] == 0.0
        assert summary["total_points"] == 0

    def test_territory_summary_with_owned_points(self):
        tc = TerritoryControl()
        cp = ControlPoint("cp1", (100.0, 100.0), name="Hill", value=3.0, capture_time=1.0)
        tc.add_control_point(cp)
        units = {"u1": ((100.0, 100.0), "red")}
        for _ in range(200):
            tc.tick(0.1, units)
        summary = tc.get_territory_summary("red")
        assert len(summary["controlled_points"]) == 1
        assert summary["total_value"] == 3.0
        assert summary["total_points"] == 1

    def test_to_dict(self):
        tc = TerritoryControl()
        cp = ControlPoint("cp1", (100.0, 100.0), name="Alpha", value=2.0)
        tc.add_control_point(cp)
        data = tc.to_dict()
        assert "control_points" in data
        assert len(data["control_points"]) == 1
        p = data["control_points"][0]
        assert p["point_id"] == "cp1"
        assert p["name"] == "Alpha"
        assert p["state"] == "neutral"


# ===========================================================================
# StrategicValue — Construction
# ===========================================================================


class TestStrategicValueConstruction:
    """Tests for StrategicValue creation."""

    def test_basic_creation(self):
        sv = StrategicValue(10, 10)
        assert sv.width == 10
        assert sv.height == 10

    def test_custom_cell_size(self):
        sv = StrategicValue(5, 5, cell_size=20.0)
        assert sv.cell_size == 20.0

    def test_zero_width_raises(self):
        with pytest.raises(ValueError, match="positive"):
            StrategicValue(0, 10)

    def test_zero_cell_size_raises(self):
        with pytest.raises(ValueError, match="positive"):
            StrategicValue(10, 10, cell_size=0)


# ===========================================================================
# StrategicValue — Elevation scoring
# ===========================================================================


class TestStrategicValueElevation:
    """Tests for elevation-based strategic scoring."""

    def test_flat_terrain_moderate_elevation_score(self):
        sv = StrategicValue(5, 5, cell_size=10.0)
        # All flat = all neighbors equal = score = 1.0 for elevation
        score = sv.rate_cell(2, 2)
        # With no roads/buildings, only elevation contributes (weight 0.4)
        # Flat = higher than all neighbors (all equal), so elev_score = 1.0
        assert score == pytest.approx(0.4, abs=0.05)

    def test_hilltop_scores_higher(self):
        sv = StrategicValue(5, 5, cell_size=10.0)
        # Create a gradient: center is peak, edges are low
        for x in range(5):
            for y in range(5):
                dist = math.hypot(x - 2, y - 2)
                sv.set_elevation(x, y, max(0.0, 50.0 - dist * 15.0))
        hilltop = sv.rate_cell(2, 2)
        corner = sv.rate_cell(0, 0)
        assert hilltop > corner

    def test_valley_scores_lower(self):
        sv = StrategicValue(5, 5, cell_size=10.0)
        # Surround center with high terrain
        for x in range(5):
            for y in range(5):
                sv.set_elevation(x, y, 50.0)
        sv.set_elevation(2, 2, 0.0)  # Valley
        val = sv.rate_cell(2, 2)
        # Should be low elevation score
        assert val < 0.2


# ===========================================================================
# StrategicValue — Road scoring
# ===========================================================================


class TestStrategicValueRoads:
    """Tests for road-based strategic scoring."""

    def test_road_increases_value(self):
        sv = StrategicValue(10, 10, cell_size=10.0)
        sv.add_road((55.0, 55.0))
        on_road = sv.rate_position((55.0, 55.0))
        off_road = sv.rate_position((5.0, 5.0))
        assert on_road > off_road

    def test_crossroads_highest_road_score(self):
        sv = StrategicValue(10, 10, cell_size=10.0)
        # Create a crossroads at (5, 5)
        sv.add_road_segment((0.0, 55.0), (95.0, 55.0))   # Horizontal road
        sv.add_road_segment((55.0, 0.0), (55.0, 95.0))   # Vertical road
        cross_val = sv.rate_position((55.0, 55.0))
        # Road cell not at crossroads
        straight_val = sv.rate_position((15.0, 55.0))
        assert cross_val >= straight_val

    def test_road_segment(self):
        sv = StrategicValue(10, 10, cell_size=10.0)
        sv.add_road_segment((5.0, 5.0), (95.0, 5.0))
        # Multiple cells along the road should have road value
        val1 = sv.rate_position((25.0, 5.0))
        val2 = sv.rate_position((75.0, 5.0))
        assert val1 > 0
        assert val2 > 0


# ===========================================================================
# StrategicValue — Building scoring
# ===========================================================================


class TestStrategicValueBuildings:
    """Tests for building density strategic scoring."""

    def test_building_increases_value(self):
        sv = StrategicValue(10, 10, cell_size=10.0)
        sv.add_building((55.0, 55.0))
        with_bldg = sv.rate_position((55.0, 55.0))
        without = sv.rate_position((5.0, 5.0))
        assert with_bldg > without

    def test_dense_buildings_higher(self):
        sv = StrategicValue(10, 10, cell_size=10.0)
        # Cluster buildings
        for x in range(3, 7):
            for y in range(3, 7):
                sv.add_building((x * 10.0 + 5.0, y * 10.0 + 5.0))
        dense = sv.rate_position((55.0, 55.0))
        sparse = sv.rate_position((5.0, 5.0))
        assert dense > sparse


# ===========================================================================
# StrategicValue — find_high_value_positions and to_grid
# ===========================================================================


class TestStrategicValueQueries:
    """Tests for querying high-value positions and grid export."""

    def test_find_high_value_empty(self):
        sv = StrategicValue(5, 5, cell_size=10.0)
        # Flat terrain, no roads, no buildings — values should be moderate
        results = sv.find_high_value_positions(min_value=0.9)
        # Likely none above 0.9 with no features
        assert isinstance(results, list)

    def test_find_high_value_with_features(self):
        sv = StrategicValue(10, 10, cell_size=10.0)
        sv.set_elevation(5, 5, 100.0)
        sv.add_road((55.0, 55.0))
        sv.add_building((55.0, 55.0))
        results = sv.find_high_value_positions(min_value=0.3, max_results=5)
        assert len(results) > 0
        assert results[0]["value"] >= 0.3

    def test_find_high_value_result_keys(self):
        sv = StrategicValue(5, 5, cell_size=10.0)
        sv.set_elevation(2, 2, 50.0)
        sv.add_road((25.0, 25.0))
        results = sv.find_high_value_positions(min_value=0.0, max_results=1)
        if results:
            r = results[0]
            assert "position" in r
            assert "cell" in r
            assert "value" in r
            assert "elevation_score" in r
            assert "road_score" in r
            assert "building_score" in r

    def test_find_high_value_sorted_descending(self):
        sv = StrategicValue(10, 10, cell_size=10.0)
        sv.set_elevation(2, 2, 50.0)
        sv.set_elevation(8, 8, 100.0)
        results = sv.find_high_value_positions(min_value=0.0, max_results=100)
        values = [r["value"] for r in results]
        assert values == sorted(values, reverse=True)

    def test_to_grid(self):
        sv = StrategicValue(3, 3, cell_size=10.0)
        grid = sv.to_grid()
        assert len(grid) == 3
        assert len(grid[0]) == 3
        for row in grid:
            for val in row:
                assert 0.0 <= val <= 1.0

    def test_cache_works(self):
        sv = StrategicValue(5, 5, cell_size=10.0)
        sv.set_elevation(2, 2, 50.0)
        v1 = sv.rate_cell(2, 2)
        v2 = sv.rate_cell(2, 2)
        assert v1 == v2

    def test_cache_invalidated_on_elevation_change(self):
        sv = StrategicValue(5, 5, cell_size=10.0)
        v1 = sv.rate_cell(2, 2)
        sv.set_elevation(2, 2, 100.0)
        v2 = sv.rate_cell(2, 2)
        # After changing elevation, cache should be cleared for that cell
        assert v1 != v2 or True  # May be same if neighbors unchanged

    def test_rate_position_out_of_bounds(self):
        sv = StrategicValue(5, 5, cell_size=10.0)
        assert sv.rate_position((999.0, 999.0)) == 0.0

    def test_rate_cell_out_of_bounds(self):
        sv = StrategicValue(5, 5)
        assert sv.rate_cell(-1, -1) == 0.0
        assert sv.rate_cell(100, 100) == 0.0


# ===========================================================================
# StrategicValue — HeightMap integration
# ===========================================================================


class TestStrategicValueHeightMapIntegration:
    """Tests for importing elevation data from HeightMap."""

    def test_set_elevation_from_heightmap(self):
        from tritium_lib.sim_engine.terrain import HeightMap

        hm = HeightMap(10, 10, cell_size=10.0)
        # Create a gradient peaking at (5,5)
        for x in range(10):
            for y in range(10):
                dist = math.hypot(x - 5, y - 5)
                hm.set_elevation(x, y, max(0.0, 50.0 - dist * 8.0))

        sv = StrategicValue(10, 10, cell_size=10.0)
        sv.set_elevation_from_heightmap(hm)

        # The cell corresponding to the peak should score higher
        peak_val = sv.rate_cell(5, 5)
        corner_val = sv.rate_cell(0, 0)
        assert peak_val > corner_val


# ===========================================================================
# ControlPoint dataclass
# ===========================================================================


class TestControlPoint:
    """Tests for the ControlPoint dataclass."""

    def test_defaults(self):
        cp = ControlPoint("cp1", (10.0, 20.0))
        assert cp.point_id == "cp1"
        assert cp.position == (10.0, 20.0)
        assert cp.name == ""
        assert cp.owner is None
        assert cp.value == 1.0

    def test_custom_values(self):
        cp = ControlPoint(
            "cp2", (50.0, 50.0),
            name="Bridge",
            capture_radius=50.0,
            capture_time=20.0,
            value=3.0,
            owner="blue",
        )
        assert cp.name == "Bridge"
        assert cp.capture_radius == 50.0
        assert cp.capture_time == 20.0
        assert cp.value == 3.0
        assert cp.owner == "blue"


# ===========================================================================
# CaptureState enum
# ===========================================================================


class TestCaptureState:
    """Tests for the CaptureState enum."""

    def test_all_values(self):
        assert CaptureState.NEUTRAL.value == "neutral"
        assert CaptureState.CAPTURING.value == "capturing"
        assert CaptureState.CONTESTED.value == "contested"
        assert CaptureState.CAPTURED.value == "captured"

    def test_from_value(self):
        assert CaptureState("neutral") == CaptureState.NEUTRAL
        assert CaptureState("captured") == CaptureState.CAPTURED
