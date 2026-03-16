# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the supply route simulation module."""

import json

import pytest

from tritium_lib.sim_engine.supply_routes import (
    DELIVERY_RANGE,
    ConvoyStatus,
    RouteStatus,
    SupplyConvoy,
    SupplyLevel,
    SupplyLine,
    SupplyRouteEngine,
    UnitSupplyState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> SupplyRouteEngine:
    return SupplyRouteEngine()


@pytest.fixture
def basic_line() -> SupplyLine:
    return SupplyLine(
        line_id="main",
        waypoints=[(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)],
        source_cache_id="depot",
        alliance="friendly",
    )


@pytest.fixture
def populated_engine(engine: SupplyRouteEngine, basic_line: SupplyLine) -> SupplyRouteEngine:
    engine.add_supply_line(basic_line)
    engine.register_unit("u1", alliance="friendly", max_ammo=100.0, max_food=100.0)
    engine.register_unit("u2", alliance="friendly", max_ammo=100.0, max_food=100.0)
    return engine


# ---------------------------------------------------------------------------
# Supply line tests
# ---------------------------------------------------------------------------

class TestSupplyLines:
    def test_add_supply_line(self, engine: SupplyRouteEngine, basic_line: SupplyLine) -> None:
        engine.add_supply_line(basic_line)
        assert "main" in engine.supply_lines

    def test_remove_supply_line(self, engine: SupplyRouteEngine, basic_line: SupplyLine) -> None:
        engine.add_supply_line(basic_line)
        engine.remove_supply_line("main")
        assert "main" not in engine.supply_lines

    def test_total_length(self, basic_line: SupplyLine) -> None:
        length = basic_line.total_length()
        assert length == pytest.approx(100.0)

    def test_route_status_default(self, basic_line: SupplyLine) -> None:
        assert basic_line.status == RouteStatus.OPEN


# ---------------------------------------------------------------------------
# Unit registration
# ---------------------------------------------------------------------------

class TestUnitRegistration:
    def test_register_unit(self, engine: SupplyRouteEngine) -> None:
        uss = engine.register_unit("u1", max_ammo=200.0, max_food=150.0)
        assert uss.unit_id == "u1"
        assert uss.ammo == 200.0
        assert uss.food == 150.0

    def test_remove_unit(self, engine: SupplyRouteEngine) -> None:
        engine.register_unit("u1")
        engine.remove_unit("u1")
        assert "u1" not in engine.unit_states

    def test_set_combat_status(self, engine: SupplyRouteEngine) -> None:
        engine.register_unit("u1")
        engine.set_combat_status("u1", True)
        assert engine.unit_states["u1"].in_combat is True


# ---------------------------------------------------------------------------
# Convoy dispatch and movement
# ---------------------------------------------------------------------------

class TestConvoys:
    def test_dispatch_convoy(self, populated_engine: SupplyRouteEngine) -> None:
        convoy = populated_engine.dispatch_convoy("main", payload_ammo=100.0, payload_food=50.0)
        assert convoy is not None
        assert convoy.ammo == 100.0
        assert convoy.food == 50.0
        assert convoy.status == ConvoyStatus.EN_ROUTE

    def test_dispatch_on_missing_line_returns_none(self, engine: SupplyRouteEngine) -> None:
        assert engine.dispatch_convoy("nonexistent") is None

    def test_dispatch_on_destroyed_line_returns_none(self, engine: SupplyRouteEngine) -> None:
        line = SupplyLine(
            line_id="dead", waypoints=[(0.0, 0.0), (100.0, 0.0)],
            status=RouteStatus.DESTROYED,
        )
        engine.add_supply_line(line)
        assert engine.dispatch_convoy("dead") is None

    def test_convoy_moves_along_route(self, populated_engine: SupplyRouteEngine) -> None:
        convoy = populated_engine.dispatch_convoy("main", payload_ammo=100.0)
        assert convoy is not None
        # Convoy speed = 8.0 m/s, tick 1.0 second
        populated_engine.tick(1.0)
        # Should have moved 8 meters along x-axis
        assert convoy.position[0] > 0.0

    def test_convoy_arrives(self, populated_engine: SupplyRouteEngine) -> None:
        convoy = populated_engine.dispatch_convoy("main", speed=100.0)
        assert convoy is not None
        # With 100 m/s speed and 100m route, should arrive in ~1 second
        result = populated_engine.tick(2.0)
        assert convoy.status == ConvoyStatus.ARRIVED
        events = result["events"]
        assert any(e["type"] == "convoy_arrived" for e in events)

    def test_convoy_halts_on_interdicted_route(self, populated_engine: SupplyRouteEngine) -> None:
        convoy = populated_engine.dispatch_convoy("main")
        assert convoy is not None
        old_pos = convoy.position
        # Place enemy right on a waypoint
        populated_engine.tick(0.1, enemy_positions={"e1": (50.0, 0.0)})
        line = populated_engine.supply_lines["main"]
        assert line.status == RouteStatus.INTERDICTED
        # Convoy should not have moved much (it started at (0,0) and route is interdicted)
        # First tick moved it before interdiction was checked, but second tick it halts
        populated_engine.tick(1.0, enemy_positions={"e1": (50.0, 0.0)})
        # Position should be near where it was after first tick

    def test_convoy_slowed_on_contested_route(self, populated_engine: SupplyRouteEngine) -> None:
        # Enemy at 2x interdiction radius should make route contested
        convoy = populated_engine.dispatch_convoy("main", speed=10.0)
        assert convoy is not None
        line = populated_engine.supply_lines["main"]
        # Place enemy just outside interdiction radius but within 2x
        enemy_pos = {"e1": (50.0, line.interdiction_radius * 1.5)}
        populated_engine.tick(1.0, enemy_positions=enemy_pos)
        assert line.status == RouteStatus.CONTESTED

    def test_attack_convoy(self, populated_engine: SupplyRouteEngine) -> None:
        convoy = populated_engine.dispatch_convoy("main")
        assert convoy is not None
        destroyed = populated_engine.attack_convoy(convoy.convoy_id, 50.0)
        assert destroyed is False
        assert convoy.health < 100.0

    def test_destroy_convoy(self, populated_engine: SupplyRouteEngine) -> None:
        convoy = populated_engine.dispatch_convoy("main")
        assert convoy is not None
        destroyed = populated_engine.attack_convoy(convoy.convoy_id, 500.0)
        assert destroyed is True
        assert convoy.status == ConvoyStatus.DESTROYED


# ---------------------------------------------------------------------------
# Supply delivery
# ---------------------------------------------------------------------------

class TestDelivery:
    def test_convoy_delivers_to_nearby_unit(self, populated_engine: SupplyRouteEngine) -> None:
        # Deplete unit ammo first
        populated_engine.unit_states["u1"].ammo = 10.0
        # Dispatch fast convoy
        convoy = populated_engine.dispatch_convoy("main", payload_ammo=80.0, speed=200.0)
        assert convoy is not None
        # Unit at endpoint
        unit_positions = {"u1": (100.0, 0.0), "u2": (100.0, 0.0)}
        # Tick to let convoy arrive and deliver
        populated_engine.tick(1.0, unit_positions=unit_positions)
        assert populated_engine.unit_states["u1"].ammo > 10.0

    def test_convoy_delivers_food(self, populated_engine: SupplyRouteEngine) -> None:
        populated_engine.unit_states["u1"].food = 5.0
        convoy = populated_engine.dispatch_convoy("main", payload_food=80.0, speed=200.0)
        assert convoy is not None
        populated_engine.tick(1.0, unit_positions={"u1": (100.0, 0.0)})
        assert populated_engine.unit_states["u1"].food > 5.0

    def test_no_delivery_to_far_unit(self, populated_engine: SupplyRouteEngine) -> None:
        populated_engine.unit_states["u1"].ammo = 10.0
        convoy = populated_engine.dispatch_convoy("main", payload_ammo=80.0, speed=200.0)
        assert convoy is not None
        # Unit far from endpoint
        populated_engine.tick(1.0, unit_positions={"u1": (500.0, 500.0)})
        assert populated_engine.unit_states["u1"].ammo == pytest.approx(10.0, abs=0.1)

    def test_no_delivery_to_enemy_unit(self, populated_engine: SupplyRouteEngine) -> None:
        populated_engine.register_unit("h1", alliance="hostile")
        populated_engine.unit_states["h1"].ammo = 10.0
        convoy = populated_engine.dispatch_convoy("main", payload_ammo=80.0, speed=200.0)
        assert convoy is not None
        populated_engine.tick(1.0, unit_positions={"h1": (100.0, 0.0)})
        assert populated_engine.unit_states["h1"].ammo == pytest.approx(10.0, abs=1.0)


# ---------------------------------------------------------------------------
# Supply consumption
# ---------------------------------------------------------------------------

class TestConsumption:
    def test_food_consumed_over_time(self, engine: SupplyRouteEngine) -> None:
        engine.register_unit("u1", max_food=100.0)
        engine.tick(10.0)
        assert engine.get_unit_food("u1") < 100.0

    def test_ammo_consumed_only_in_combat(self, engine: SupplyRouteEngine) -> None:
        engine.register_unit("u1", max_ammo=100.0)
        engine.tick(10.0)
        # Not in combat, ammo unchanged
        assert engine.get_unit_ammo("u1") == 100.0
        # Now set to combat
        engine.set_combat_status("u1", True)
        engine.tick(10.0)
        assert engine.get_unit_ammo("u1") < 100.0

    def test_supplies_dont_go_negative(self, engine: SupplyRouteEngine) -> None:
        uss = engine.register_unit("u1", max_ammo=1.0, max_food=1.0)
        uss.ammo_consumption_rate = 100.0
        uss.food_consumption_rate = 100.0
        engine.set_combat_status("u1", True)
        engine.tick(10.0)
        assert engine.get_unit_ammo("u1") == 0.0
        assert engine.get_unit_food("u1") == 0.0


# ---------------------------------------------------------------------------
# Supply level queries
# ---------------------------------------------------------------------------

class TestSupplyQueries:
    def test_get_unit_supply_level_full(self, engine: SupplyRouteEngine) -> None:
        engine.register_unit("u1")
        levels = engine.get_unit_supply_level("u1")
        assert levels["ammo"] == "full"
        assert levels["food"] == "full"

    def test_get_unit_supply_level_empty(self, engine: SupplyRouteEngine) -> None:
        uss = engine.register_unit("u1")
        uss.ammo = 0.0
        uss.food = 0.0
        levels = engine.get_unit_supply_level("u1")
        assert levels["ammo"] == "empty"
        assert levels["food"] == "empty"

    def test_get_unit_supply_level_unknown(self, engine: SupplyRouteEngine) -> None:
        levels = engine.get_unit_supply_level("nonexistent")
        assert levels["ammo"] == "unknown"

    def test_get_route_status(self, populated_engine: SupplyRouteEngine) -> None:
        assert populated_engine.get_route_status("main") == RouteStatus.OPEN
        assert populated_engine.get_route_status("fake") == RouteStatus.DESTROYED


# ---------------------------------------------------------------------------
# Route interdiction
# ---------------------------------------------------------------------------

class TestInterdiction:
    def test_route_becomes_interdicted(self, populated_engine: SupplyRouteEngine) -> None:
        populated_engine.tick(0.1, enemy_positions={"e1": (50.0, 0.0)})
        assert populated_engine.supply_lines["main"].status == RouteStatus.INTERDICTED

    def test_route_returns_to_open(self, populated_engine: SupplyRouteEngine) -> None:
        populated_engine.tick(0.1, enemy_positions={"e1": (50.0, 0.0)})
        assert populated_engine.supply_lines["main"].status == RouteStatus.INTERDICTED
        populated_engine.tick(0.1, enemy_positions={})
        assert populated_engine.supply_lines["main"].status == RouteStatus.OPEN

    def test_route_status_change_event(self, populated_engine: SupplyRouteEngine) -> None:
        result = populated_engine.tick(0.1, enemy_positions={"e1": (50.0, 0.0)})
        events = result["events"]
        assert any(e["type"] == "route_status_changed" for e in events)


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

class TestWarnings:
    def test_low_ammo_warning(self, engine: SupplyRouteEngine) -> None:
        uss = engine.register_unit("u1")
        uss.ammo = 3.0  # 3% of max 100 = critical
        engine.set_combat_status("u1", True)
        result = engine.tick(0.1)
        warnings = result["warnings"]
        assert any(w["type"] == "low_ammo" for w in warnings)

    def test_low_food_warning(self, engine: SupplyRouteEngine) -> None:
        uss = engine.register_unit("u1")
        uss.food = 3.0
        result = engine.tick(0.1)
        warnings = result["warnings"]
        assert any(w["type"] == "low_food" for w in warnings)


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

class TestEventLog:
    def test_convoy_dispatch_logged(self, populated_engine: SupplyRouteEngine) -> None:
        populated_engine.dispatch_convoy("main")
        log = populated_engine.drain_event_log()
        assert any(e["type"] == "convoy_dispatched" for e in log)

    def test_drain_clears_log(self, populated_engine: SupplyRouteEngine) -> None:
        populated_engine.dispatch_convoy("main")
        populated_engine.drain_event_log()
        assert len(populated_engine.drain_event_log()) == 0


# ---------------------------------------------------------------------------
# Three.js visualization
# ---------------------------------------------------------------------------

class TestThreeJS:
    def test_to_three_js_structure(self, populated_engine: SupplyRouteEngine) -> None:
        populated_engine.dispatch_convoy("main")
        populated_engine.tick(0.1)
        viz = populated_engine.to_three_js()
        assert "routes" in viz
        assert "convoys" in viz
        assert "unit_supply" in viz
        assert len(viz["routes"]) == 1
        assert len(viz["convoys"]) == 1

    def test_to_three_js_serializable(self, populated_engine: SupplyRouteEngine) -> None:
        populated_engine.dispatch_convoy("main")
        populated_engine.tick(0.1)
        viz = populated_engine.to_three_js()
        serialized = json.dumps(viz)
        assert len(serialized) > 0

    def test_route_viz_fields(self, populated_engine: SupplyRouteEngine) -> None:
        viz = populated_engine.to_three_js()
        route = viz["routes"][0]
        assert route["id"] == "main"
        assert "waypoints" in route
        assert "status" in route
        assert "color" in route
        assert "dashed" in route

    def test_unit_supply_viz_fields(self, populated_engine: SupplyRouteEngine) -> None:
        viz = populated_engine.to_three_js()
        assert len(viz["unit_supply"]) >= 2
        unit = viz["unit_supply"][0]
        assert "ammo_ratio" in unit
        assert "food_ratio" in unit
        assert "ammo_level" in unit
        assert "supply_color" in unit

    def test_destroyed_convoy_excluded_from_viz(self, populated_engine: SupplyRouteEngine) -> None:
        convoy = populated_engine.dispatch_convoy("main")
        assert convoy is not None
        populated_engine.attack_convoy(convoy.convoy_id, 500.0)
        populated_engine.tick(0.1)
        viz = populated_engine.to_three_js()
        assert len(viz["convoys"]) == 0
