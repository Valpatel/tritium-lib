# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the economy and resource management system."""

from __future__ import annotations

import pytest

from tritium_lib.sim_engine.economy import (
    BuildQueue,
    EconomyEngine,
    ECONOMY_PRESETS,
    ResourcePool,
    ResourceType,
    TechTree,
    TECH_TREE,
    UnitCost,
    UNIT_COSTS,
    _RESOURCE_COLORS,
    _build_default_tech_tree,
)

RT = ResourceType  # shorthand


# ═══════════════════════════════════════════════════════════════════════════
# ResourceType enum
# ═══════════════════════════════════════════════════════════════════════════


class TestResourceType:
    def test_all_members(self):
        assert len(ResourceType) == 6

    def test_values(self):
        assert RT.CREDITS.value == "credits"
        assert RT.MANPOWER.value == "manpower"
        assert RT.FUEL.value == "fuel"
        assert RT.STEEL.value == "steel"
        assert RT.ELECTRONICS.value == "electronics"
        assert RT.FOOD.value == "food"


# ═══════════════════════════════════════════════════════════════════════════
# ResourcePool
# ═══════════════════════════════════════════════════════════════════════════


class TestResourcePool:
    def _pool(self, **kwargs) -> ResourcePool:
        return ResourcePool(
            resources={RT.CREDITS: 100, RT.FUEL: 50},
            income={RT.CREDITS: 10, RT.FUEL: 5},
            capacity={RT.CREDITS: 200, RT.FUEL: 100},
            **kwargs,
        )

    def test_get(self):
        p = self._pool()
        assert p.get(RT.CREDITS) == 100
        assert p.get(RT.STEEL) == 0  # not present

    def test_add_within_capacity(self):
        p = self._pool()
        added = p.add(RT.CREDITS, 50)
        assert added == 50
        assert p.get(RT.CREDITS) == 150

    def test_add_capped_by_capacity(self):
        p = self._pool()
        added = p.add(RT.CREDITS, 200)
        assert added == 100  # capacity is 200, already at 100
        assert p.get(RT.CREDITS) == 200

    def test_add_zero_or_negative(self):
        p = self._pool()
        assert p.add(RT.CREDITS, 0) == 0
        assert p.add(RT.CREDITS, -10) == 0

    def test_add_no_capacity(self):
        p = ResourcePool(resources={RT.CREDITS: 100})
        added = p.add(RT.CREDITS, 999999)
        assert added == 999999  # no cap = infinite

    def test_spend(self):
        p = self._pool()
        spent = p.spend(RT.CREDITS, 30)
        assert spent == 30
        assert p.get(RT.CREDITS) == 70

    def test_spend_more_than_available(self):
        p = self._pool()
        spent = p.spend(RT.CREDITS, 200)
        assert spent == 100
        assert p.get(RT.CREDITS) == 0

    def test_spend_zero_or_negative(self):
        p = self._pool()
        assert p.spend(RT.CREDITS, 0) == 0
        assert p.spend(RT.CREDITS, -5) == 0

    def test_can_afford_true(self):
        p = self._pool()
        assert p.can_afford({RT.CREDITS: 50, RT.FUEL: 50})

    def test_can_afford_false(self):
        p = self._pool()
        assert not p.can_afford({RT.CREDITS: 101})

    def test_can_afford_empty_cost(self):
        p = self._pool()
        assert p.can_afford({})

    def test_spend_exact_success(self):
        p = self._pool()
        ok = p.spend_exact({RT.CREDITS: 50, RT.FUEL: 25})
        assert ok
        assert p.get(RT.CREDITS) == 50
        assert p.get(RT.FUEL) == 25

    def test_spend_exact_fail_atomic(self):
        p = self._pool()
        ok = p.spend_exact({RT.CREDITS: 50, RT.FUEL: 999})
        assert not ok
        # Nothing should have been spent
        assert p.get(RT.CREDITS) == 100
        assert p.get(RT.FUEL) == 50

    def test_tick_applies_income(self):
        p = self._pool()
        p.tick(1.0)
        assert p.get(RT.CREDITS) == 110
        assert p.get(RT.FUEL) == 55

    def test_tick_caps_at_capacity(self):
        p = ResourcePool(
            resources={RT.CREDITS: 195},
            income={RT.CREDITS: 10},
            capacity={RT.CREDITS: 200},
        )
        p.tick(1.0)
        assert p.get(RT.CREDITS) == 200

    def test_tick_dt_scaling(self):
        p = self._pool()
        p.tick(0.5)
        assert p.get(RT.CREDITS) == 105  # 10 * 0.5

    def test_tick_zero_dt(self):
        p = self._pool()
        p.tick(0.0)
        assert p.get(RT.CREDITS) == 100

    def test_snapshot(self):
        p = self._pool()
        s = p.snapshot()
        assert s["credits"] == 100
        assert s["fuel"] == 50


# ═══════════════════════════════════════════════════════════════════════════
# UnitCost
# ═══════════════════════════════════════════════════════════════════════════


class TestUnitCost:
    def test_creation(self):
        uc = UnitCost(
            unit_template="infantry",
            cost={RT.CREDITS: 50},
            build_time=5.0,
            upkeep={RT.CREDITS: 2},
        )
        assert uc.unit_template == "infantry"
        assert uc.build_time == 5.0
        assert uc.cost[RT.CREDITS] == 50
        assert uc.upkeep[RT.CREDITS] == 2

    def test_defaults(self):
        uc = UnitCost(unit_template="test")
        assert uc.build_time == 10.0
        assert uc.cost == {}
        assert uc.upkeep == {}


# ═══════════════════════════════════════════════════════════════════════════
# BuildQueue
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildQueue:
    def test_empty(self):
        bq = BuildQueue()
        assert len(bq) == 0
        assert bq.peek() is None
        assert bq.pop_completed() == []

    def test_add_and_peek(self):
        bq = BuildQueue()
        bq.add("infantry", {RT.CREDITS: 50}, 5.0)
        assert len(bq) == 1
        item = bq.peek()
        assert item is not None
        assert item["unit_template"] == "infantry"
        assert item["progress"] == 0.0

    def test_tick_advances_first_only(self):
        bq = BuildQueue()
        bq.add("infantry", {RT.CREDITS: 50}, 5.0)
        bq.add("sniper", {RT.CREDITS: 120}, 10.0)
        bq.tick(3.0)
        assert bq.queue[0]["progress"] == 3.0
        assert bq.queue[1]["progress"] == 0.0

    def test_pop_completed(self):
        bq = BuildQueue()
        bq.add("infantry", {RT.CREDITS: 50}, 5.0)
        bq.tick(6.0)
        done = bq.pop_completed()
        assert done == ["infantry"]
        assert len(bq) == 0

    def test_pop_completed_partial(self):
        bq = BuildQueue()
        bq.add("infantry", {}, 2.0)
        bq.add("sniper", {}, 10.0)
        bq.tick(3.0)  # first finishes
        done = bq.pop_completed()
        assert done == ["infantry"]
        assert len(bq) == 1

    def test_cancel(self):
        bq = BuildQueue()
        bq.add("infantry", {}, 5.0)
        bq.add("sniper", {}, 10.0)
        cancelled = bq.cancel(0)
        assert cancelled is not None
        assert cancelled["unit_template"] == "infantry"
        assert len(bq) == 1

    def test_cancel_out_of_range(self):
        bq = BuildQueue()
        assert bq.cancel(0) is None

    def test_to_list(self):
        bq = BuildQueue()
        bq.add("infantry", {RT.CREDITS: 50}, 10.0)
        bq.tick(5.0)
        lst = bq.to_list()
        assert len(lst) == 1
        assert lst[0]["unit_template"] == "infantry"
        assert lst[0]["progress_pct"] == 0.5
        assert lst[0]["remaining_s"] == 5.0

    def test_serial_production(self):
        """Second item only starts building after first completes."""
        bq = BuildQueue()
        bq.add("infantry", {}, 3.0)
        bq.add("scout", {}, 2.0)
        bq.tick(3.0)
        bq.pop_completed()
        bq.tick(2.0)
        done = bq.pop_completed()
        assert done == ["scout"]
        assert len(bq) == 0


# ═══════════════════════════════════════════════════════════════════════════
# TechTree
# ═══════════════════════════════════════════════════════════════════════════


class TestTechTree:
    def _tree(self) -> TechTree:
        tree = TechTree()
        tree.add_tech("t1", {RT.CREDITS: 100}, unlocks=["infantry"])
        tree.add_tech("t2", {RT.CREDITS: 200}, prerequisites=["t1"], unlocks=["tank"])
        tree.add_tech("t3", {RT.CREDITS: 300}, prerequisites=["t1", "t2"], unlocks=["heli"])
        return tree

    def test_available_no_prereqs(self):
        tree = self._tree()
        assert tree.is_available("t1")
        assert not tree.is_available("t2")

    def test_available_after_research(self):
        tree = self._tree()
        pool = ResourcePool(resources={RT.CREDITS: 999})
        tree.research("t1", pool)
        assert tree.is_available("t2")
        assert not tree.is_available("t3")  # needs t2 too

    def test_research_spends_resources(self):
        tree = self._tree()
        pool = ResourcePool(resources={RT.CREDITS: 150})
        ok = tree.research("t1", pool)
        assert ok
        assert pool.get(RT.CREDITS) == 50

    def test_research_cannot_afford(self):
        tree = self._tree()
        pool = ResourcePool(resources={RT.CREDITS: 10})
        assert not tree.research("t1", pool)
        assert pool.get(RT.CREDITS) == 10  # unchanged

    def test_research_unavailable(self):
        tree = self._tree()
        pool = ResourcePool(resources={RT.CREDITS: 999})
        assert not tree.research("t2", pool)

    def test_research_already_done(self):
        tree = self._tree()
        pool = ResourcePool(resources={RT.CREDITS: 999})
        tree.research("t1", pool)
        assert not tree.research("t1", pool)  # already researched

    def test_research_unknown_tech(self):
        tree = self._tree()
        pool = ResourcePool(resources={RT.CREDITS: 999})
        assert not tree.research("nonexistent", pool)

    def test_get_unlocked_units(self):
        tree = self._tree()
        pool = ResourcePool(resources={RT.CREDITS: 999})
        tree.research("t1", pool)
        assert "infantry" in tree.get_unlocked_units()
        assert "tank" not in tree.get_unlocked_units()

    def test_available_techs(self):
        tree = self._tree()
        assert tree.available_techs() == ["t1"]

    def test_to_dict(self):
        tree = self._tree()
        d = tree.to_dict()
        assert "t1" in d
        assert d["t1"]["available"] is True
        assert d["t2"]["available"] is False
        assert d["t1"]["researched"] is False

    def test_multi_prereq(self):
        tree = self._tree()
        pool = ResourcePool(resources={RT.CREDITS: 9999})
        tree.research("t1", pool)
        tree.research("t2", pool)
        assert tree.is_available("t3")
        ok = tree.research("t3", pool)
        assert ok
        assert "heli" in tree.get_unlocked_units()


# ═══════════════════════════════════════════════════════════════════════════
# EconomyEngine
# ═══════════════════════════════════════════════════════════════════════════


class TestEconomyEngine:
    def _engine(self) -> EconomyEngine:
        engine = EconomyEngine()
        engine.setup_faction("alpha", ECONOMY_PRESETS["standard"])
        engine.register_unit_costs(UNIT_COSTS)
        engine.register_tech_tree("alpha", _build_default_tech_tree())
        return engine

    # -- setup --

    def test_setup_faction(self):
        engine = EconomyEngine()
        engine.setup_faction("test", ECONOMY_PRESETS["standard"])
        assert "test" in engine.pools
        assert "test" in engine.build_queues

    def test_setup_faction_no_preset(self):
        engine = EconomyEngine()
        engine.setup_faction("bare")
        assert engine.pools["bare"].get(RT.CREDITS) == 0

    def test_register_unit_costs(self):
        engine = EconomyEngine()
        engine.register_unit_costs(UNIT_COSTS)
        assert "infantry" in engine.unit_costs

    # -- purchase --

    def test_purchase_unit_success(self):
        engine = self._engine()
        ok = engine.purchase_unit("alpha", "infantry")
        assert ok
        assert len(engine.build_queues["alpha"]) == 1

    def test_purchase_unit_deducts_resources(self):
        engine = self._engine()
        before = engine.pools["alpha"].get(RT.CREDITS)
        engine.purchase_unit("alpha", "infantry")
        after = engine.pools["alpha"].get(RT.CREDITS)
        assert after == before - UNIT_COSTS["infantry"].cost[RT.CREDITS]

    def test_purchase_unit_insufficient_resources(self):
        engine = EconomyEngine()
        engine.setup_faction("poor")
        engine.register_unit_costs(UNIT_COSTS)
        ok = engine.purchase_unit("poor", "tank")
        assert not ok
        assert len(engine.build_queues["poor"]) == 0

    def test_purchase_unknown_template(self):
        engine = self._engine()
        assert not engine.purchase_unit("alpha", "nonexistent")

    def test_purchase_unknown_faction(self):
        engine = self._engine()
        assert not engine.purchase_unit("nobody", "infantry")

    # -- tick --

    def test_tick_income(self):
        engine = self._engine()
        before = engine.pools["alpha"].get(RT.CREDITS)
        engine.tick(1.0)
        after = engine.pools["alpha"].get(RT.CREDITS)
        expected_income = ECONOMY_PRESETS["standard"]["income"][RT.CREDITS]
        assert after == pytest.approx(before + expected_income, abs=0.01)

    def test_tick_build_progress(self):
        engine = self._engine()
        engine.purchase_unit("alpha", "infantry")
        engine.tick(3.0)
        item = engine.build_queues["alpha"].peek()
        assert item is not None
        assert item["progress"] == pytest.approx(3.0)

    def test_tick_completes_build(self):
        engine = self._engine()
        engine.purchase_unit("alpha", "infantry")
        bt = UNIT_COSTS["infantry"].build_time
        completed = engine.tick(bt + 1.0)
        assert "alpha:infantry" in completed
        assert len(engine.build_queues["alpha"]) == 0

    def test_tick_adds_to_active_units(self):
        engine = self._engine()
        engine.purchase_unit("alpha", "scout")
        bt = UNIT_COSTS["scout"].build_time
        engine.tick(bt + 1.0)
        assert "scout" in engine._active_units["alpha"]

    def test_tick_upkeep(self):
        engine = self._engine()
        engine.add_active_unit("alpha", "infantry")
        before = engine.pools["alpha"].get(RT.CREDITS)
        engine.tick(60.0)  # 60 seconds = 1 minute
        after = engine.pools["alpha"].get(RT.CREDITS)
        # Income - upkeep should net something
        income = ECONOMY_PRESETS["standard"]["income"][RT.CREDITS] * 60
        upkeep = UNIT_COSTS["infantry"].upkeep[RT.CREDITS]  # per minute, 60s = 1 min
        expected = before + income - upkeep
        assert after == pytest.approx(expected, abs=0.1)

    def test_tick_no_factions(self):
        engine = EconomyEngine()
        result = engine.tick(1.0)
        assert result == []

    # -- active units --

    def test_add_remove_active_unit(self):
        engine = self._engine()
        engine.add_active_unit("alpha", "infantry")
        assert "infantry" in engine._active_units["alpha"]
        ok = engine.remove_active_unit("alpha", "infantry")
        assert ok
        assert "infantry" not in engine._active_units["alpha"]

    def test_remove_active_unit_not_found(self):
        engine = self._engine()
        assert not engine.remove_active_unit("alpha", "tank")

    # -- tech --

    def test_research_tech(self):
        engine = self._engine()
        ok = engine.research_tech("alpha", "basic_training")
        assert ok
        assert "basic_training" in engine.tech_trees["alpha"].researched

    def test_research_tech_no_tree(self):
        engine = EconomyEngine()
        engine.setup_faction("bare")
        assert not engine.research_tech("bare", "basic_training")

    def test_research_tech_no_pool(self):
        engine = EconomyEngine()
        assert not engine.research_tech("nobody", "basic_training")

    # -- get_economy_status --

    def test_get_economy_status(self):
        engine = self._engine()
        engine.purchase_unit("alpha", "infantry")
        status = engine.get_economy_status("alpha")
        assert status["faction"] == "alpha"
        assert "credits" in status["resources"]
        assert status["build_queue_length"] == 1
        assert isinstance(status["tech_available"], list)

    def test_get_economy_status_unknown_faction(self):
        engine = self._engine()
        status = engine.get_economy_status("nobody")
        assert status["resources"] == {}

    # -- to_three_js --

    def test_to_three_js(self):
        engine = self._engine()
        engine.purchase_unit("alpha", "infantry")
        engine.add_active_unit("alpha", "infantry")
        js = engine.to_three_js("alpha")
        assert js["faction"] == "alpha"
        assert isinstance(js["resource_bars"], list)
        assert len(js["resource_bars"]) > 0
        bar = js["resource_bars"][0]
        assert "resource" in bar
        assert "value" in bar
        assert "fill" in bar
        assert "color" in bar

    def test_to_three_js_upkeep(self):
        engine = self._engine()
        engine.add_active_unit("alpha", "drone")
        js = engine.to_three_js("alpha")
        assert "upkeep_per_min" in js
        assert len(js["upkeep_per_min"]) > 0

    def test_to_three_js_unknown_faction(self):
        engine = self._engine()
        js = engine.to_three_js("nobody")
        assert js["resource_bars"] == []


# ═══════════════════════════════════════════════════════════════════════════
# Presets and data tables
# ═══════════════════════════════════════════════════════════════════════════


class TestPresets:
    def test_all_economy_presets_valid(self):
        for name, preset in ECONOMY_PRESETS.items():
            assert "resources" in preset, f"preset {name} missing resources"
            assert "income" in preset, f"preset {name} missing income"
            assert "capacity" in preset, f"preset {name} missing capacity"
            # All keys should be ResourceType
            for key in preset["resources"]:
                assert isinstance(key, ResourceType)

    def test_preset_count(self):
        assert len(ECONOMY_PRESETS) >= 5

    def test_standard_preset_has_all_resources(self):
        p = ECONOMY_PRESETS["standard"]
        for rt in ResourceType:
            assert rt in p["resources"]
            assert rt in p["income"]
            assert rt in p["capacity"]

    def test_unit_costs_defined(self):
        assert len(UNIT_COSTS) >= 8

    def test_unit_costs_all_have_templates(self):
        for name, uc in UNIT_COSTS.items():
            assert uc.unit_template == name
            assert uc.build_time > 0

    def test_unit_costs_infantry(self):
        uc = UNIT_COSTS["infantry"]
        assert RT.CREDITS in uc.cost
        assert RT.MANPOWER in uc.cost
        assert uc.build_time == 5.0

    def test_unit_costs_tank_expensive(self):
        assert UNIT_COSTS["tank"].cost[RT.CREDITS] > UNIT_COSTS["infantry"].cost[RT.CREDITS]

    def test_resource_colors_all_types(self):
        for rt in ResourceType:
            assert rt in _RESOURCE_COLORS


class TestDefaultTechTree:
    def test_tech_count(self):
        tree = _build_default_tech_tree()
        assert len(tree.techs) >= 15

    def test_tier1_no_prereqs(self):
        tree = _build_default_tech_tree()
        tier1 = ["basic_training", "steel_works", "electronics_lab",
                  "logistics_hub", "field_rations"]
        for t in tier1:
            assert t in tree.techs
            assert tree.techs[t]["prerequisites"] == []

    def test_tier2_has_prereqs(self):
        tree = _build_default_tech_tree()
        assert "electronics_lab" in tree.techs["drone_warfare"]["prerequisites"]
        assert "steel_works" in tree.techs["mechanized_infantry"]["prerequisites"]

    def test_tier3_has_tier2_prereqs(self):
        tree = _build_default_tech_tree()
        assert "mechanized_infantry" in tree.techs["armored_warfare"]["prerequisites"]
        assert "drone_warfare" in tree.techs["air_superiority"]["prerequisites"]

    def test_full_research_path(self):
        """Research from tier 1 all the way to tier 3 tank."""
        tree = _build_default_tech_tree()
        pool = ResourcePool(resources={rt: 99999 for rt in ResourceType})
        # Tier 1
        assert tree.research("steel_works", pool)
        assert tree.research("logistics_hub", pool)
        # Tier 2
        assert tree.research("mechanized_infantry", pool)
        # Tier 3
        assert tree.research("armored_warfare", pool)
        assert "tank" in tree.get_unlocked_units()

    def test_all_techs_have_cost(self):
        tree = _build_default_tech_tree()
        for name, tech in tree.techs.items():
            assert RT.CREDITS in tech["cost"], f"{name} missing CREDITS cost"

    def test_singleton_is_fresh(self):
        """TECH_TREE module-level instance should have nothing researched."""
        assert len(TECH_TREE.researched) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Integration / multi-faction
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiFaction:
    def test_two_factions(self):
        engine = EconomyEngine()
        engine.setup_faction("alpha", ECONOMY_PRESETS["standard"])
        engine.setup_faction("bravo", ECONOMY_PRESETS["skirmish"])
        engine.register_unit_costs(UNIT_COSTS)
        assert engine.pools["alpha"].get(RT.CREDITS) > engine.pools["bravo"].get(RT.CREDITS)

    def test_independent_purchases(self):
        engine = EconomyEngine()
        engine.setup_faction("alpha", ECONOMY_PRESETS["rich"])
        engine.setup_faction("bravo", ECONOMY_PRESETS["rich"])
        engine.register_unit_costs(UNIT_COSTS)
        engine.purchase_unit("alpha", "tank")
        assert len(engine.build_queues["alpha"]) == 1
        assert len(engine.build_queues["bravo"]) == 0

    def test_tick_both_factions(self):
        engine = EconomyEngine()
        engine.setup_faction("a", ECONOMY_PRESETS["standard"])
        engine.setup_faction("b", ECONOMY_PRESETS["standard"])
        engine.register_unit_costs(UNIT_COSTS)
        engine.purchase_unit("a", "infantry")
        engine.purchase_unit("b", "scout")
        bt = max(UNIT_COSTS["infantry"].build_time, UNIT_COSTS["scout"].build_time)
        completed = engine.tick(bt + 1)
        assert "a:infantry" in completed
        assert "b:scout" in completed

    def test_faction_isolation_resources(self):
        engine = EconomyEngine()
        engine.setup_faction("a", ECONOMY_PRESETS["standard"])
        engine.setup_faction("b", ECONOMY_PRESETS["standard"])
        engine.pools["a"].spend(RT.CREDITS, 500)
        assert engine.pools["b"].get(RT.CREDITS) == 1000  # unaffected


class TestEdgeCases:
    def test_empty_engine_status(self):
        engine = EconomyEngine()
        s = engine.get_economy_status("nobody")
        assert s["faction"] == "nobody"
        assert s["resources"] == {}

    def test_tick_zero_dt(self):
        engine = EconomyEngine()
        engine.setup_faction("a", ECONOMY_PRESETS["standard"])
        before = engine.pools["a"].get(RT.CREDITS)
        engine.tick(0.0)
        assert engine.pools["a"].get(RT.CREDITS) == before

    def test_purchase_multiple_queued(self):
        engine = EconomyEngine()
        engine.setup_faction("a", ECONOMY_PRESETS["rich"])
        engine.register_unit_costs(UNIT_COSTS)
        for _ in range(5):
            engine.purchase_unit("a", "infantry")
        assert len(engine.build_queues["a"]) == 5

    def test_upkeep_drains_to_zero(self):
        """Upkeep should not drive resources negative."""
        engine = EconomyEngine()
        engine.setup_faction("a", {
            "resources": {RT.CREDITS: 1},
            "income": {},
            "capacity": {RT.CREDITS: 100},
        })
        engine.register_unit_costs(UNIT_COSTS)
        engine.add_active_unit("a", "infantry")
        engine.tick(600.0)  # 10 minutes of upkeep
        assert engine.pools["a"].get(RT.CREDITS) >= 0

    def test_build_queue_to_list_empty(self):
        bq = BuildQueue()
        assert bq.to_list() == []

    def test_resource_pool_snapshot_empty(self):
        p = ResourcePool()
        assert p.snapshot() == {}
