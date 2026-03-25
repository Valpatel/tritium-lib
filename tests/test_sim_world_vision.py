# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.world.vision — VisionSystem."""

import math
import pytest
from types import SimpleNamespace

from tritium_lib.sim_engine.world.vision import (
    VisionSystem,
    VisibilityState,
    SightingReport,
)
from tritium_lib.sim_engine.core.spatial import SpatialGrid


def _make_target(tid, x, y, alliance="friendly", asset_type="rover",
                 status="active", heading=0.0):
    ident = SimpleNamespace(
        bluetooth_mac=None, wifi_mac=None, cell_id=None,
    )
    return SimpleNamespace(
        target_id=tid,
        position=(x, y),
        alliance=alliance,
        asset_type=asset_type,
        status=status,
        heading=heading,
        identity=ident,
    )


def _make_radio_target(tid, x, y, alliance="hostile", bt_mac="AA:BB:CC:DD:EE:FF"):
    ident = SimpleNamespace(
        bluetooth_mac=bt_mac, wifi_mac=None, cell_id=None,
    )
    return SimpleNamespace(
        target_id=tid,
        position=(x, y),
        alliance=alliance,
        asset_type="person",
        status="active",
        heading=0.0,
        identity=ident,
    )


class TestVisibilityState:
    def test_empty_state(self):
        state = VisibilityState()
        assert state.visible_to == {}
        assert state.can_see == {}
        assert state.friendly_visible == set()
        assert state.drone_relayed == set()
        assert state.radio_detected == set()


class TestVisionSystemConstruction:
    def test_basic_creation(self):
        vs = VisionSystem()
        assert vs._terrain_map is None
        assert vs._last_state is None

    def test_with_terrain(self):
        terrain = SimpleNamespace(line_of_sight=lambda a, b: True)
        vs = VisionSystem(terrain_map=terrain)
        assert vs._terrain_map is terrain

    def test_set_terrain(self):
        vs = VisionSystem()
        terrain = SimpleNamespace(line_of_sight=lambda a, b: True)
        vs.set_terrain(terrain)
        assert vs._terrain_map is terrain


class TestVisionSystemTick:
    def test_friendly_sees_nearby_hostile(self):
        vs = VisionSystem()
        grid = SpatialGrid(cell_size=50.0)
        friendly = _make_target("f1", 0.0, 0.0, "friendly")
        hostile = _make_target("h1", 5.0, 5.0, "hostile")
        targets = {"f1": friendly, "h1": hostile}
        grid.rebuild(list(targets.values()))

        state = vs.tick(0.1, targets, grid)
        assert "h1" in state.friendly_visible

    def test_friendly_cannot_see_distant_hostile(self):
        vs = VisionSystem()
        grid = SpatialGrid(cell_size=50.0)
        friendly = _make_target("f1", 0.0, 0.0, "friendly")
        hostile = _make_target("h1", 500.0, 500.0, "hostile")
        targets = {"f1": friendly, "h1": hostile}
        grid.rebuild(list(targets.values()))

        state = vs.tick(0.1, targets, grid)
        assert "h1" not in state.friendly_visible

    def test_destroyed_targets_invisible(self):
        vs = VisionSystem()
        grid = SpatialGrid(cell_size=50.0)
        friendly = _make_target("f1", 0.0, 0.0, "friendly")
        hostile = _make_target("h1", 5.0, 5.0, "hostile", status="destroyed")
        targets = {"f1": friendly, "h1": hostile}
        grid.rebuild(list(targets.values()))

        state = vs.tick(0.1, targets, grid)
        assert "h1" not in state.friendly_visible

    def test_eliminated_targets_invisible(self):
        vs = VisionSystem()
        grid = SpatialGrid(cell_size=50.0)
        friendly = _make_target("f1", 0.0, 0.0, "friendly")
        hostile = _make_target("h1", 5.0, 5.0, "hostile", status="eliminated")
        targets = {"f1": friendly, "h1": hostile}
        grid.rebuild(list(targets.values()))

        state = vs.tick(0.1, targets, grid)
        assert "h1" not in state.friendly_visible

    def test_destroyed_observer_cannot_see(self):
        vs = VisionSystem()
        grid = SpatialGrid(cell_size=50.0)
        friendly = _make_target("f1", 0.0, 0.0, "friendly", status="destroyed")
        hostile = _make_target("h1", 5.0, 5.0, "hostile")
        targets = {"f1": friendly, "h1": hostile}
        grid.rebuild(list(targets.values()))

        state = vs.tick(0.1, targets, grid)
        assert "h1" not in state.friendly_visible

    def test_can_see_query(self):
        vs = VisionSystem()
        grid = SpatialGrid(cell_size=50.0)
        friendly = _make_target("f1", 0.0, 0.0, "friendly")
        hostile = _make_target("h1", 5.0, 5.0, "hostile")
        targets = {"f1": friendly, "h1": hostile}
        grid.rebuild(list(targets.values()))

        vs.tick(0.1, targets, grid)
        assert vs.can_see("f1", "h1")
        assert not vs.can_see("f1", "nonexistent")

    def test_can_see_before_tick_returns_false(self):
        vs = VisionSystem()
        assert not vs.can_see("f1", "h1")


class TestVisionSystemExternalSightings:
    def test_external_sighting_makes_visible(self):
        vs = VisionSystem()
        grid = SpatialGrid(cell_size=50.0)
        hostile = _make_target("h1", 500.0, 500.0, "hostile")
        targets = {"h1": hostile}
        grid.rebuild(list(targets.values()))

        report = SightingReport(
            observer_id="camera_1", target_id="h1",
            observer_type="camera", confidence=0.9,
        )
        vs.add_sighting(report)
        state = vs.tick(0.1, targets, grid)
        assert "h1" in state.friendly_visible

    def test_external_sightings_cleared_after_tick(self):
        vs = VisionSystem()
        grid = SpatialGrid(cell_size=50.0)
        hostile = _make_target("h1", 500.0, 500.0, "hostile")
        targets = {"h1": hostile}
        grid.rebuild(list(targets.values()))

        vs.add_sighting(SightingReport("cam1", "h1"))
        vs.tick(0.1, targets, grid)

        # Second tick without adding sighting — should NOT persist
        state2 = vs.tick(0.1, targets, grid)
        # h1 is still far away, so without sighting it shouldn't be visible
        assert "h1" not in state2.friendly_visible


class TestVisionSystemRadioDetection:
    def test_radio_detection_through_walls(self):
        vs = VisionSystem()
        grid = SpatialGrid(cell_size=50.0)
        friendly = _make_target("f1", 0.0, 0.0, "friendly")
        hostile = _make_radio_target("h1", 50.0, 0.0, "hostile")
        targets = {"f1": friendly, "h1": hostile}
        grid.rebuild(list(targets.values()))

        state = vs.tick(0.1, targets, grid)
        assert "h1" in state.radio_detected
        assert state.radio_signal_strength.get("h1", 0) > 0

    def test_no_radio_detection_without_mac(self):
        vs = VisionSystem()
        grid = SpatialGrid(cell_size=50.0)
        friendly = _make_target("f1", 0.0, 0.0, "friendly")
        # Hostile with no radio signatures
        hostile = _make_target("h1", 50.0, 0.0, "hostile")
        targets = {"f1": friendly, "h1": hostile}
        grid.rebuild(list(targets.values()))

        state = vs.tick(0.1, targets, grid)
        assert "h1" not in state.radio_detected

    def test_radio_signal_strength_decreases_with_distance(self):
        vs = VisionSystem()
        grid = SpatialGrid(cell_size=200.0)
        friendly = _make_target("f1", 0.0, 0.0, "friendly")
        close_hostile = _make_radio_target("h_close", 10.0, 0.0)
        far_hostile = _make_radio_target("h_far", 80.0, 0.0)
        targets = {"f1": friendly, "h_close": close_hostile, "h_far": far_hostile}
        grid.rebuild(list(targets.values()))

        state = vs.tick(0.1, targets, grid)
        close_strength = state.radio_signal_strength.get("h_close", 0)
        far_strength = state.radio_signal_strength.get("h_far", 0)
        assert close_strength > far_strength


class TestVisionSystemDroneRelay:
    def test_drone_relays_sightings_to_friendlies(self):
        vs = VisionSystem()
        grid = SpatialGrid(cell_size=50.0)
        drone = _make_target("d1", 0.0, 0.0, "friendly", asset_type="drone")
        hostile = _make_target("h1", 5.0, 5.0, "hostile")
        targets = {"d1": drone, "h1": hostile}
        grid.rebuild(list(targets.values()))

        state = vs.tick(0.1, targets, grid)
        assert "h1" in state.drone_relayed
        assert "h1" in state.friendly_visible


class TestVisionSystemLifecycle:
    def test_remove_unit_clears_sweep_angle(self):
        vs = VisionSystem()
        vs._sweep_angles["u1"] = 45.0
        vs.remove_unit("u1")
        assert "u1" not in vs._sweep_angles

    def test_reset_clears_all(self):
        vs = VisionSystem()
        vs._sweep_angles["u1"] = 45.0
        vs._sweep_angles["u2"] = 90.0
        vs.reset()
        assert len(vs._sweep_angles) == 0
