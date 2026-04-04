# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine.world.hazards — environmental hazard management."""

import pytest
from unittest.mock import MagicMock

from tritium_lib.sim_engine.world.hazards import (
    Hazard,
    HazardManager,
    HAZARD_TYPES,
)


class TestHazard:
    """Tests for the Hazard dataclass."""

    def test_defaults(self):
        h = Hazard(
            id="h1", hazard_type="fire",
            position=(10.0, 20.0), radius=5.0, duration=30.0,
        )
        assert h.active is True
        assert h.elapsed == 0.0


class TestHazardManager:
    """Tests for HazardManager lifecycle and queries."""

    def test_spawn_hazard(self):
        bus = MagicMock()
        mgr = HazardManager(event_bus=bus)
        h = mgr.spawn_hazard("fire", (10.0, 20.0), 5.0, 30.0)
        assert h.hazard_type == "fire"
        assert h.active is True
        assert len(mgr.active_hazards) == 1
        bus.publish.assert_called_once_with("hazard_spawned", {
            "id": h.id,
            "hazard_type": "fire",
            "position": {"x": 10.0, "y": 20.0},
            "radius": 5.0,
            "duration": 30.0,
        })

    def test_spawn_random(self):
        mgr = HazardManager()
        hazards = mgr.spawn_random(5, 100.0)
        assert len(hazards) == 5
        assert len(mgr.active_hazards) == 5
        for h in hazards:
            assert h.hazard_type in HAZARD_TYPES
            assert -100.0 <= h.position[0] <= 100.0
            assert -100.0 <= h.position[1] <= 100.0

    def test_tick_expires_hazards(self):
        bus = MagicMock()
        mgr = HazardManager(event_bus=bus)
        h = mgr.spawn_hazard("roadblock", (0.0, 0.0), 10.0, 5.0)
        # Advance past duration
        mgr.tick(6.0)
        assert len(mgr.active_hazards) == 0
        # Should have published both spawned and expired events
        assert bus.publish.call_count == 2
        expired_call = bus.publish.call_args_list[1]
        assert expired_call[0][0] == "hazard_expired"

    def test_is_blocked_inside(self):
        mgr = HazardManager()
        mgr.spawn_hazard("fire", (10.0, 10.0), 5.0, 30.0)
        assert mgr.is_blocked((10.0, 10.0)) is True
        assert mgr.is_blocked((12.0, 10.0)) is True

    def test_is_blocked_outside(self):
        mgr = HazardManager()
        mgr.spawn_hazard("fire", (10.0, 10.0), 5.0, 30.0)
        assert mgr.is_blocked((100.0, 100.0)) is False

    def test_get_blocked_nodes(self):
        mgr = HazardManager()
        mgr.spawn_hazard("fire", (10.0, 20.0), 5.0, 30.0)
        mgr.spawn_hazard("flood", (30.0, 40.0), 8.0, 30.0)
        nodes = mgr.get_blocked_nodes()
        assert len(nodes) == 2
        assert (10.0, 20.0) in nodes
        assert (30.0, 40.0) in nodes

    def test_clear(self):
        mgr = HazardManager()
        mgr.spawn_random(3, 50.0)
        assert len(mgr.active_hazards) == 3
        mgr.clear()
        assert len(mgr.active_hazards) == 0

    def test_to_telemetry(self):
        mgr = HazardManager()
        mgr.spawn_hazard("roadblock", (5.0, 10.0), 7.0, 20.0)
        telem = mgr.to_telemetry()
        assert len(telem) == 1
        assert telem[0]["hazard_type"] == "roadblock"
        assert telem[0]["position"] == {"x": 5.0, "y": 10.0}
        assert telem[0]["radius"] == 7.0
        assert telem[0]["active"] is True
        assert telem[0]["remaining"] == 20.0

    def test_no_event_bus(self):
        """Manager works without event bus."""
        mgr = HazardManager(event_bus=None)
        h = mgr.spawn_hazard("fire", (0.0, 0.0), 5.0, 1.0)
        assert h.active
        mgr.tick(2.0)
        assert len(mgr.active_hazards) == 0

    def test_partial_tick(self):
        """Hazards accumulate elapsed time across multiple ticks."""
        mgr = HazardManager()
        mgr.spawn_hazard("fire", (0.0, 0.0), 5.0, 10.0)
        mgr.tick(3.0)
        assert len(mgr.active_hazards) == 1
        mgr.tick(3.0)
        assert len(mgr.active_hazards) == 1
        mgr.tick(5.0)  # Total: 11s > 10s duration
        assert len(mgr.active_hazards) == 0
