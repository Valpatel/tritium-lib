"""Tests for the supply and logistics system.

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import time

import pytest

from tritium_lib.sim_engine.logistics import (
    DEFAULT_RESUPPLY_RANGE,
    LOW_SUPPLY_THRESHOLD,
    SUPPLY_PRESETS,
    LogisticsEngine,
    LowSupplyWarning,
    SupplyCache,
    SupplyRequest,
    SupplyRoute,
    SupplyType,
    cache_from_preset,
)


# ---------------------------------------------------------------------------
# SupplyType enum
# ---------------------------------------------------------------------------


class TestSupplyType:
    def test_all_members(self):
        names = {m.name for m in SupplyType}
        assert names == {"AMMO", "FUEL", "FOOD", "MEDICAL", "PARTS", "WATER"}

    def test_values(self):
        assert SupplyType.AMMO.value == "ammo"
        assert SupplyType.FUEL.value == "fuel"
        assert SupplyType.FOOD.value == "food"
        assert SupplyType.MEDICAL.value == "medical"
        assert SupplyType.PARTS.value == "parts"
        assert SupplyType.WATER.value == "water"

    def test_member_count(self):
        assert len(SupplyType) == 6


# ---------------------------------------------------------------------------
# SupplyCache
# ---------------------------------------------------------------------------


def _full_cache(cache_id: str = "c1", pos: tuple = (50.0, 50.0)) -> SupplyCache:
    return SupplyCache(
        cache_id=cache_id,
        position=pos,
        supplies={SupplyType.AMMO: 100.0, SupplyType.FUEL: 200.0},
        capacity={SupplyType.AMMO: 100.0, SupplyType.FUEL: 200.0},
        alliance="friendly",
    )


class TestSupplyCache:
    def test_available(self):
        c = _full_cache()
        assert c.available(SupplyType.AMMO) == 100.0
        assert c.available(SupplyType.WATER) == 0.0

    def test_available_destroyed(self):
        c = _full_cache()
        c.is_destroyed = True
        assert c.available(SupplyType.AMMO) == 0.0

    def test_withdraw_basic(self):
        c = _full_cache()
        taken = c.withdraw(SupplyType.AMMO, 30.0)
        assert taken == 30.0
        assert c.supplies[SupplyType.AMMO] == 70.0

    def test_withdraw_more_than_available(self):
        c = _full_cache()
        taken = c.withdraw(SupplyType.AMMO, 150.0)
        assert taken == 100.0
        assert c.supplies[SupplyType.AMMO] == 0.0

    def test_withdraw_zero(self):
        c = _full_cache()
        assert c.withdraw(SupplyType.AMMO, 0.0) == 0.0

    def test_withdraw_negative(self):
        c = _full_cache()
        assert c.withdraw(SupplyType.AMMO, -10.0) == 0.0

    def test_withdraw_destroyed(self):
        c = _full_cache()
        c.is_destroyed = True
        assert c.withdraw(SupplyType.AMMO, 10.0) == 0.0

    def test_deposit_basic(self):
        c = _full_cache()
        c.supplies[SupplyType.AMMO] = 50.0
        added = c.deposit(SupplyType.AMMO, 30.0)
        assert added == 30.0
        assert c.supplies[SupplyType.AMMO] == 80.0

    def test_deposit_over_capacity(self):
        c = _full_cache()
        c.supplies[SupplyType.AMMO] = 90.0
        added = c.deposit(SupplyType.AMMO, 20.0)
        assert added == 10.0
        assert c.supplies[SupplyType.AMMO] == 100.0

    def test_deposit_destroyed(self):
        c = _full_cache()
        c.is_destroyed = True
        assert c.deposit(SupplyType.AMMO, 10.0) == 0.0

    def test_deposit_zero(self):
        c = _full_cache()
        assert c.deposit(SupplyType.AMMO, 0.0) == 0.0

    def test_fill_ratio_full(self):
        c = _full_cache()
        assert c.fill_ratio(SupplyType.AMMO) == 1.0

    def test_fill_ratio_half(self):
        c = _full_cache()
        c.supplies[SupplyType.AMMO] = 50.0
        assert c.fill_ratio(SupplyType.AMMO) == 0.5

    def test_fill_ratio_empty(self):
        c = _full_cache()
        c.supplies[SupplyType.AMMO] = 0.0
        assert c.fill_ratio(SupplyType.AMMO) == 0.0

    def test_fill_ratio_no_capacity(self):
        c = _full_cache()
        assert c.fill_ratio(SupplyType.WATER) == 0.0

    def test_total_fill_ratio_full(self):
        c = _full_cache()
        assert c.total_fill_ratio() == 1.0

    def test_total_fill_ratio_mixed(self):
        c = _full_cache()
        c.supplies[SupplyType.AMMO] = 50.0  # 50%
        # FUEL still at 200 = 100%
        assert c.total_fill_ratio() == pytest.approx(0.75)

    def test_total_fill_ratio_no_capacity(self):
        c = SupplyCache(cache_id="e", position=(0, 0))
        assert c.total_fill_ratio() == 0.0


# ---------------------------------------------------------------------------
# SupplyRequest
# ---------------------------------------------------------------------------


class TestSupplyRequest:
    def test_defaults(self):
        r = SupplyRequest(requester_id="u1", supply_type=SupplyType.AMMO, amount=50.0)
        assert r.priority == 1
        assert r.fulfilled is False
        assert r.timestamp > 0

    def test_custom_fields(self):
        r = SupplyRequest(
            requester_id="u2",
            supply_type=SupplyType.FUEL,
            amount=100.0,
            priority=3,
            position=(10.0, 20.0),
            timestamp=1234.0,
        )
        assert r.priority == 3
        assert r.position == (10.0, 20.0)
        assert r.timestamp == 1234.0


# ---------------------------------------------------------------------------
# SupplyRoute
# ---------------------------------------------------------------------------


class TestSupplyRoute:
    def test_defaults(self):
        r = SupplyRoute(route_id="r1")
        assert r.is_active is True
        assert r.risk_level == 0.0
        assert r.waypoints == []

    def test_custom(self):
        r = SupplyRoute(
            route_id="r2",
            waypoints=[(0, 0), (50, 50)],
            source_cache_id="c1",
            dest_cache_id="c2",
            risk_level=0.7,
        )
        assert len(r.waypoints) == 2
        assert r.risk_level == 0.7


# ---------------------------------------------------------------------------
# SUPPLY_PRESETS & cache_from_preset
# ---------------------------------------------------------------------------


class TestPresets:
    def test_all_presets_exist(self):
        expected = {"infantry_fob", "vehicle_depot", "field_hospital", "ammo_dump", "forward_cache"}
        assert set(SUPPLY_PRESETS.keys()) == expected

    def test_infantry_fob(self):
        p = SUPPLY_PRESETS["infantry_fob"]
        assert p[SupplyType.AMMO] == 1000.0
        assert p[SupplyType.MEDICAL] == 200.0
        assert p[SupplyType.FOOD] == 500.0
        assert p[SupplyType.WATER] == 500.0

    def test_vehicle_depot(self):
        p = SUPPLY_PRESETS["vehicle_depot"]
        assert p[SupplyType.FUEL] == 5000.0
        assert p[SupplyType.PARTS] == 500.0

    def test_cache_from_preset_full(self):
        c = cache_from_preset("ammo_dump", "ad1", (10.0, 20.0), "friendly")
        assert c.supplies[SupplyType.AMMO] == 5000.0
        assert c.capacity[SupplyType.AMMO] == 5000.0
        assert c.alliance == "friendly"

    def test_cache_from_preset_half_fill(self):
        c = cache_from_preset("ammo_dump", "ad2", (0, 0), fill=0.5)
        assert c.supplies[SupplyType.AMMO] == 2500.0
        assert c.capacity[SupplyType.AMMO] == 5000.0

    def test_cache_from_preset_empty(self):
        c = cache_from_preset("forward_cache", "fc1", (0, 0), fill=0.0)
        assert c.supplies[SupplyType.AMMO] == 0.0
        assert c.capacity[SupplyType.AMMO] == 200.0

    def test_cache_from_preset_invalid(self):
        with pytest.raises(KeyError):
            cache_from_preset("nonexistent", "x", (0, 0))


# ---------------------------------------------------------------------------
# LogisticsEngine — basic operations
# ---------------------------------------------------------------------------


class TestLogisticsEngineBasic:
    def test_add_cache(self):
        eng = LogisticsEngine()
        c = _full_cache()
        eng.add_cache(c)
        assert "c1" in eng.caches

    def test_remove_cache(self):
        eng = LogisticsEngine()
        eng.add_cache(_full_cache())
        removed = eng.remove_cache("c1")
        assert removed is not None
        assert "c1" not in eng.caches

    def test_remove_missing_cache(self):
        eng = LogisticsEngine()
        assert eng.remove_cache("nope") is None

    def test_add_route(self):
        eng = LogisticsEngine()
        eng.add_route(SupplyRoute(route_id="r1", waypoints=[(0, 0), (100, 100)]))
        assert len(eng.routes) == 1

    def test_request_supply(self):
        eng = LogisticsEngine()
        req = eng.request_supply("u1", SupplyType.AMMO, 50.0, priority=2, position=(10, 10))
        assert len(eng.requests) == 1
        assert req.priority == 2
        assert req.fulfilled is False

    def test_resupply_unit(self):
        eng = LogisticsEngine()
        eng.add_cache(_full_cache())
        taken = eng.resupply_unit("u1", "c1", SupplyType.AMMO, 30.0)
        assert taken == 30.0
        assert eng.unit_supplies["u1"][SupplyType.AMMO] == 30.0
        assert eng.caches["c1"].supplies[SupplyType.AMMO] == 70.0

    def test_resupply_unit_limited_by_stock(self):
        eng = LogisticsEngine()
        eng.add_cache(_full_cache())
        taken = eng.resupply_unit("u1", "c1", SupplyType.AMMO, 200.0)
        assert taken == 100.0

    def test_resupply_missing_cache(self):
        eng = LogisticsEngine()
        assert eng.resupply_unit("u1", "nope", SupplyType.AMMO, 10.0) == 0.0

    def test_resupply_destroyed_cache(self):
        eng = LogisticsEngine()
        c = _full_cache()
        c.is_destroyed = True
        eng.add_cache(c)
        assert eng.resupply_unit("u1", "c1", SupplyType.AMMO, 10.0) == 0.0

    def test_resupply_accumulates(self):
        eng = LogisticsEngine()
        eng.add_cache(_full_cache())
        eng.resupply_unit("u1", "c1", SupplyType.AMMO, 20.0)
        eng.resupply_unit("u1", "c1", SupplyType.AMMO, 15.0)
        assert eng.unit_supplies["u1"][SupplyType.AMMO] == 35.0


# ---------------------------------------------------------------------------
# LogisticsEngine — find_nearest_cache
# ---------------------------------------------------------------------------


class TestFindNearestCache:
    def test_basic(self):
        eng = LogisticsEngine()
        eng.add_cache(_full_cache("c1", (0.0, 0.0)))
        eng.add_cache(_full_cache("c2", (100.0, 0.0)))
        nearest = eng.find_nearest_cache((10.0, 0.0), "friendly")
        assert nearest is not None
        assert nearest.cache_id == "c1"

    def test_filters_alliance(self):
        eng = LogisticsEngine()
        c = _full_cache("c1", (0.0, 0.0))
        c.alliance = "hostile"
        eng.add_cache(c)
        assert eng.find_nearest_cache((0.0, 0.0), "friendly") is None

    def test_filters_destroyed(self):
        eng = LogisticsEngine()
        c = _full_cache("c1", (0.0, 0.0))
        c.is_destroyed = True
        eng.add_cache(c)
        assert eng.find_nearest_cache((0.0, 0.0), "friendly") is None

    def test_filters_supply_type(self):
        eng = LogisticsEngine()
        c = _full_cache("c1", (0.0, 0.0))
        c.supplies[SupplyType.AMMO] = 0.0
        eng.add_cache(c)
        assert eng.find_nearest_cache((0.0, 0.0), "friendly", SupplyType.AMMO) is None
        # FUEL is still available
        assert eng.find_nearest_cache((0.0, 0.0), "friendly", SupplyType.FUEL) is not None

    def test_no_caches(self):
        eng = LogisticsEngine()
        assert eng.find_nearest_cache((0.0, 0.0), "friendly") is None

    def test_picks_closer(self):
        eng = LogisticsEngine()
        eng.add_cache(_full_cache("far", (1000.0, 0.0)))
        eng.add_cache(_full_cache("near", (5.0, 0.0)))
        nearest = eng.find_nearest_cache((0.0, 0.0), "friendly")
        assert nearest is not None
        assert nearest.cache_id == "near"


# ---------------------------------------------------------------------------
# LogisticsEngine — get_supply_status
# ---------------------------------------------------------------------------


class TestGetSupplyStatus:
    def test_basic(self):
        eng = LogisticsEngine()
        eng.add_cache(cache_from_preset("ammo_dump", "a1", (0, 0), "friendly"))
        status = eng.get_supply_status("friendly")
        assert status["alliance"] == "friendly"
        assert status["cache_count"] == 1
        assert status["destroyed_count"] == 0
        assert status["totals"]["ammo"] == 5000.0

    def test_destroyed_excluded(self):
        eng = LogisticsEngine()
        c = cache_from_preset("ammo_dump", "a1", (0, 0), "friendly")
        c.is_destroyed = True
        eng.add_cache(c)
        status = eng.get_supply_status("friendly")
        assert status["cache_count"] == 0
        assert status["destroyed_count"] == 1

    def test_low_supply_detected(self):
        eng = LogisticsEngine()
        c = cache_from_preset("ammo_dump", "a1", (0, 0), "friendly", fill=0.1)
        eng.add_cache(c)
        status = eng.get_supply_status("friendly")
        assert len(status["low_supply"]) == 1
        assert status["low_supply"][0]["supply_type"] == "ammo"

    def test_filters_alliance(self):
        eng = LogisticsEngine()
        eng.add_cache(cache_from_preset("ammo_dump", "a1", (0, 0), "friendly"))
        eng.add_cache(cache_from_preset("ammo_dump", "a2", (0, 0), "hostile"))
        status = eng.get_supply_status("hostile")
        assert status["cache_count"] == 1

    def test_ratios(self):
        eng = LogisticsEngine()
        c = cache_from_preset("ammo_dump", "a1", (0, 0), "friendly", fill=0.5)
        eng.add_cache(c)
        status = eng.get_supply_status("friendly")
        assert status["ratios"]["ammo"] == 0.5


# ---------------------------------------------------------------------------
# LogisticsEngine — tick (consumption)
# ---------------------------------------------------------------------------


class TestTickConsumption:
    def test_consumes_supplies(self):
        eng = LogisticsEngine()
        eng.set_unit_supplies("u1", {SupplyType.AMMO: 100.0})
        eng.set_consumption_rate("u1", {SupplyType.AMMO: 10.0})  # 10/sec
        eng.tick(1.0, {"u1": (0.0, 0.0)})
        assert eng.unit_supplies["u1"][SupplyType.AMMO] == pytest.approx(90.0)

    def test_consumption_cannot_go_negative(self):
        eng = LogisticsEngine()
        eng.set_unit_supplies("u1", {SupplyType.AMMO: 5.0})
        eng.set_consumption_rate("u1", {SupplyType.AMMO: 10.0})
        eng.tick(1.0, {"u1": (0.0, 0.0)})
        assert eng.unit_supplies["u1"][SupplyType.AMMO] == pytest.approx(0.0)

    def test_no_supplies_no_crash(self):
        eng = LogisticsEngine()
        eng.set_consumption_rate("u1", {SupplyType.AMMO: 10.0})
        # u1 has no unit_supplies entry -- should not crash
        eng.tick(1.0, {"u1": (0.0, 0.0)})

    def test_multi_supply_consumption(self):
        eng = LogisticsEngine()
        eng.set_unit_supplies("u1", {SupplyType.AMMO: 100.0, SupplyType.FUEL: 50.0})
        eng.set_consumption_rate("u1", {SupplyType.AMMO: 5.0, SupplyType.FUEL: 2.0})
        eng.tick(2.0, {"u1": (0.0, 0.0)})
        assert eng.unit_supplies["u1"][SupplyType.AMMO] == pytest.approx(90.0)
        assert eng.unit_supplies["u1"][SupplyType.FUEL] == pytest.approx(46.0)


# ---------------------------------------------------------------------------
# LogisticsEngine — tick (auto-resupply)
# ---------------------------------------------------------------------------


class TestTickAutoResupply:
    def test_auto_resupply_when_low(self):
        eng = LogisticsEngine(resupply_range=100.0)
        eng.add_cache(_full_cache("c1", (0.0, 0.0)))
        # Unit has low ammo near cache
        eng.set_unit_supplies("u1", {SupplyType.AMMO: 1.0})
        eng.set_consumption_rate("u1", {SupplyType.AMMO: 10.0})
        eng.tick(0.1, {"u1": (10.0, 0.0)})
        # Should have been resupplied
        assert eng.unit_supplies["u1"][SupplyType.AMMO] > 1.0

    def test_no_resupply_out_of_range(self):
        eng = LogisticsEngine(resupply_range=10.0)
        eng.add_cache(_full_cache("c1", (0.0, 0.0)))
        eng.set_unit_supplies("u1", {SupplyType.AMMO: 1.0})
        eng.set_consumption_rate("u1", {SupplyType.AMMO: 10.0})
        eng.tick(0.0, {"u1": (100.0, 0.0)})
        assert eng.unit_supplies["u1"][SupplyType.AMMO] == 1.0

    def test_no_resupply_wrong_alliance(self):
        eng = LogisticsEngine(resupply_range=100.0)
        c = _full_cache("c1", (0.0, 0.0))
        c.alliance = "hostile"
        eng.add_cache(c)
        eng.set_unit_supplies("u1", {SupplyType.AMMO: 1.0})
        eng.set_consumption_rate("u1", {SupplyType.AMMO: 10.0})
        eng.tick(0.0, {"u1": (5.0, 0.0)}, unit_alliances={"u1": "friendly"})
        assert eng.unit_supplies["u1"][SupplyType.AMMO] == 1.0


# ---------------------------------------------------------------------------
# LogisticsEngine — tick (warnings)
# ---------------------------------------------------------------------------


class TestTickWarnings:
    def test_low_supply_warning(self):
        eng = LogisticsEngine()
        c = cache_from_preset("ammo_dump", "a1", (0, 0), "friendly", fill=0.1)
        eng.add_cache(c)
        warnings = eng.tick(0.0, {})
        assert len(warnings) >= 1
        assert warnings[0].cache_id == "a1"
        assert warnings[0].supply_type == SupplyType.AMMO

    def test_no_warning_when_full(self):
        eng = LogisticsEngine()
        eng.add_cache(cache_from_preset("ammo_dump", "a1", (0, 0), "friendly", fill=1.0))
        warnings = eng.tick(0.0, {})
        assert len(warnings) == 0

    def test_destroyed_no_warning(self):
        eng = LogisticsEngine()
        c = cache_from_preset("ammo_dump", "a1", (0, 0), "friendly", fill=0.1)
        c.is_destroyed = True
        eng.add_cache(c)
        warnings = eng.tick(0.0, {})
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# LogisticsEngine — pending_requests
# ---------------------------------------------------------------------------


class TestPendingRequests:
    def test_sorted_by_priority(self):
        eng = LogisticsEngine()
        eng.request_supply("u1", SupplyType.AMMO, 50.0, priority=1)
        eng.request_supply("u2", SupplyType.AMMO, 50.0, priority=3)
        eng.request_supply("u3", SupplyType.AMMO, 50.0, priority=2)
        pending = eng.pending_requests()
        assert [r.requester_id for r in pending] == ["u2", "u3", "u1"]

    def test_fulfilled_excluded(self):
        eng = LogisticsEngine()
        req = eng.request_supply("u1", SupplyType.AMMO, 50.0)
        req.fulfilled = True
        assert len(eng.pending_requests()) == 0
        assert len(eng.pending_requests(fulfilled=True)) == 1


# ---------------------------------------------------------------------------
# LogisticsEngine — to_three_js
# ---------------------------------------------------------------------------


class TestToThreeJs:
    def test_structure(self):
        eng = LogisticsEngine()
        eng.add_cache(cache_from_preset("ammo_dump", "a1", (10.0, 20.0), "friendly"))
        eng.add_route(
            SupplyRoute(
                route_id="r1",
                waypoints=[(10, 20), (50, 60)],
                source_cache_id="a1",
                dest_cache_id="a2",
                risk_level=0.3,
            )
        )
        eng.request_supply("u1", SupplyType.AMMO, 50.0, position=(100, 80))
        out = eng.to_three_js()

        assert "caches" in out
        assert "routes" in out
        assert "requests" in out

        assert len(out["caches"]) == 1
        c = out["caches"][0]
        assert c["id"] == "a1"
        assert c["x"] == 10.0
        assert c["y"] == 20.0
        assert c["alliance"] == "friendly"
        assert c["color"] == "#05ffa1"
        assert "ammo" in c["supplies"]

        assert len(out["routes"]) == 1
        r = out["routes"][0]
        assert r["id"] == "r1"
        assert r["risk"] == 0.3
        assert r["active"] is True
        assert len(r["waypoints"]) == 2

        assert len(out["requests"]) == 1
        rq = out["requests"][0]
        assert rq["type"] == "ammo"
        assert rq["priority"] == 1

    def test_hostile_color(self):
        eng = LogisticsEngine()
        c = _full_cache("h1", (0, 0))
        c.alliance = "hostile"
        eng.add_cache(c)
        out = eng.to_three_js()
        assert out["caches"][0]["color"] == "#ff2a6d"

    def test_neutral_color(self):
        eng = LogisticsEngine()
        c = _full_cache("n1", (0, 0))
        c.alliance = "neutral"
        eng.add_cache(c)
        out = eng.to_three_js()
        assert out["caches"][0]["color"] == "#fcee0a"

    def test_fulfilled_requests_excluded(self):
        eng = LogisticsEngine()
        req = eng.request_supply("u1", SupplyType.AMMO, 50.0)
        req.fulfilled = True
        out = eng.to_three_js()
        assert len(out["requests"]) == 0

    def test_destroyed_cache_included_with_flag(self):
        eng = LogisticsEngine()
        c = _full_cache("d1", (0, 0))
        c.is_destroyed = True
        eng.add_cache(c)
        out = eng.to_three_js()
        assert out["caches"][0]["destroyed"] is True

    def test_supply_ratios_in_output(self):
        eng = LogisticsEngine()
        c = _full_cache("c1", (0, 0))
        c.supplies[SupplyType.AMMO] = 50.0  # half
        eng.add_cache(c)
        out = eng.to_three_js()
        assert out["caches"][0]["supplies"]["ammo"] == 0.5

    def test_empty_engine(self):
        eng = LogisticsEngine()
        out = eng.to_three_js()
        assert out == {"caches": [], "routes": [], "requests": []}


# ---------------------------------------------------------------------------
# Integration / multi-tick scenarios
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_resupply_cycle(self):
        """Unit consumes, gets resupplied, consumes more."""
        eng = LogisticsEngine(resupply_range=100.0)
        eng.add_cache(cache_from_preset("ammo_dump", "a1", (0, 0), "friendly"))
        eng.set_unit_supplies("u1", {SupplyType.AMMO: 100.0})
        eng.set_consumption_rate("u1", {SupplyType.AMMO: 50.0})  # 50/sec

        # Tick 1: consume 50
        eng.tick(1.0, {"u1": (5.0, 0.0)})
        ammo_after_1 = eng.unit_supplies["u1"][SupplyType.AMMO]
        # Should have been resupplied since 50 < 50*60*0.5
        assert ammo_after_1 > 0

    def test_cache_depletion(self):
        """Cache runs out, unit stops getting resupply."""
        eng = LogisticsEngine(resupply_range=100.0)
        small = SupplyCache(
            cache_id="s1",
            position=(0, 0),
            supplies={SupplyType.AMMO: 10.0},
            capacity={SupplyType.AMMO: 10.0},
            alliance="friendly",
        )
        eng.add_cache(small)
        eng.set_unit_supplies("u1", {SupplyType.AMMO: 0.0})
        eng.set_consumption_rate("u1", {SupplyType.AMMO: 1.0})

        # Tick to trigger resupply
        eng.tick(0.1, {"u1": (1.0, 0.0)})
        # Cache should lose some ammo
        assert eng.caches["s1"].supplies[SupplyType.AMMO] < 10.0

    def test_multiple_units_share_cache(self):
        eng = LogisticsEngine(resupply_range=100.0)
        eng.add_cache(_full_cache("c1", (0, 0)))
        eng.set_unit_supplies("u1", {SupplyType.AMMO: 1.0})
        eng.set_unit_supplies("u2", {SupplyType.AMMO: 1.0})
        eng.set_consumption_rate("u1", {SupplyType.AMMO: 10.0})
        eng.set_consumption_rate("u2", {SupplyType.AMMO: 10.0})
        eng.tick(0.0, {"u1": (1.0, 0.0), "u2": (2.0, 0.0)})
        # Both should have been resupplied from same cache
        total_withdrawn = 100.0 - eng.caches["c1"].supplies[SupplyType.AMMO]
        assert total_withdrawn > 0

    def test_request_fulfilled_by_tick(self):
        eng = LogisticsEngine(resupply_range=100.0)
        eng.add_cache(_full_cache("c1", (0, 0)))
        eng.request_supply("u1", SupplyType.AMMO, 30.0, position=(5.0, 0.0))
        eng.tick(0.0, {"u1": (5.0, 0.0)})
        fulfilled = eng.pending_requests(fulfilled=True)
        assert len(fulfilled) == 1

    def test_request_not_fulfilled_out_of_range(self):
        eng = LogisticsEngine(resupply_range=10.0)
        eng.add_cache(_full_cache("c1", (0, 0)))
        eng.request_supply("u1", SupplyType.AMMO, 30.0, position=(500.0, 0.0))
        eng.tick(0.0, {"u1": (500.0, 0.0)})
        pending = eng.pending_requests(fulfilled=False)
        assert len(pending) == 1
