# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Standalone unit tests for economy.py subsystems.

Tests ResourcePool, BuildQueue, TechTree, and EconomyEngine in isolation
without requiring game_server.py or build_full_game().
"""

from __future__ import annotations

import copy

import pytest

from tritium_lib.sim_engine.economy import (
    ResourceType,
    ResourcePool,
    UnitCost,
    BuildQueue,
    TechTree,
    EconomyEngine,
    UNIT_COSTS,
    TECH_TREE,
    ECONOMY_PRESETS,
)


# ---------------------------------------------------------------------------
# ResourcePool
# ---------------------------------------------------------------------------

class TestResourcePool:
    def test_add_within_capacity(self) -> None:
        pool = ResourcePool(
            resources={ResourceType.CREDITS: 100},
            capacity={ResourceType.CREDITS: 500},
        )
        actual = pool.add(ResourceType.CREDITS, 200)
        assert actual == 200
        assert pool.get(ResourceType.CREDITS) == 300

    def test_add_capped_at_capacity(self) -> None:
        pool = ResourcePool(
            resources={ResourceType.CREDITS: 400},
            capacity={ResourceType.CREDITS: 500},
        )
        actual = pool.add(ResourceType.CREDITS, 200)
        assert actual == 100  # only 100 room
        assert pool.get(ResourceType.CREDITS) == 500

    def test_add_zero_or_negative(self) -> None:
        pool = ResourcePool(resources={ResourceType.CREDITS: 100})
        assert pool.add(ResourceType.CREDITS, 0) == 0.0
        assert pool.add(ResourceType.CREDITS, -10) == 0.0
        assert pool.get(ResourceType.CREDITS) == 100

    def test_add_no_capacity_limit(self) -> None:
        pool = ResourcePool(resources={ResourceType.CREDITS: 100})
        # No capacity set -> inf
        actual = pool.add(ResourceType.CREDITS, 999999)
        assert actual == 999999

    def test_spend_within_balance(self) -> None:
        pool = ResourcePool(resources={ResourceType.CREDITS: 100})
        actual = pool.spend(ResourceType.CREDITS, 60)
        assert actual == 60
        assert pool.get(ResourceType.CREDITS) == 40

    def test_spend_more_than_balance(self) -> None:
        pool = ResourcePool(resources={ResourceType.CREDITS: 50})
        actual = pool.spend(ResourceType.CREDITS, 100)
        assert actual == 50  # only spends what's available
        assert pool.get(ResourceType.CREDITS) == 0

    def test_spend_zero_or_negative(self) -> None:
        pool = ResourcePool(resources={ResourceType.CREDITS: 100})
        assert pool.spend(ResourceType.CREDITS, 0) == 0.0
        assert pool.spend(ResourceType.CREDITS, -5) == 0.0

    def test_spend_exact_success(self) -> None:
        pool = ResourcePool(resources={
            ResourceType.CREDITS: 100,
            ResourceType.STEEL: 50,
        })
        cost = {ResourceType.CREDITS: 80, ResourceType.STEEL: 30}
        assert pool.spend_exact(cost) is True
        assert pool.get(ResourceType.CREDITS) == 20
        assert pool.get(ResourceType.STEEL) == 20

    def test_spend_exact_failure_no_side_effects(self) -> None:
        pool = ResourcePool(resources={
            ResourceType.CREDITS: 100,
            ResourceType.STEEL: 10,  # not enough steel
        })
        cost = {ResourceType.CREDITS: 80, ResourceType.STEEL: 30}
        assert pool.spend_exact(cost) is False
        # Nothing should be deducted
        assert pool.get(ResourceType.CREDITS) == 100
        assert pool.get(ResourceType.STEEL) == 10

    def test_can_afford_true(self) -> None:
        pool = ResourcePool(resources={ResourceType.CREDITS: 500})
        assert pool.can_afford({ResourceType.CREDITS: 500}) is True

    def test_can_afford_false(self) -> None:
        pool = ResourcePool(resources={ResourceType.CREDITS: 499})
        assert pool.can_afford({ResourceType.CREDITS: 500}) is False

    def test_tick_applies_income(self) -> None:
        pool = ResourcePool(
            resources={ResourceType.CREDITS: 100},
            income={ResourceType.CREDITS: 10},
            capacity={ResourceType.CREDITS: 1000},
        )
        pool.tick(5.0)
        assert pool.get(ResourceType.CREDITS) == 150  # 100 + 10*5

    def test_tick_respects_capacity(self) -> None:
        pool = ResourcePool(
            resources={ResourceType.CREDITS: 990},
            income={ResourceType.CREDITS: 10},
            capacity={ResourceType.CREDITS: 1000},
        )
        pool.tick(5.0)
        assert pool.get(ResourceType.CREDITS) == 1000  # capped

    def test_snapshot(self) -> None:
        pool = ResourcePool(resources={
            ResourceType.CREDITS: 123.456,
            ResourceType.FUEL: 50.0,
        })
        snap = pool.snapshot()
        assert snap["credits"] == 123.46
        assert snap["fuel"] == 50.0

    def test_get_missing_resource(self) -> None:
        pool = ResourcePool()
        assert pool.get(ResourceType.CREDITS) == 0.0


# ---------------------------------------------------------------------------
# BuildQueue
# ---------------------------------------------------------------------------

class TestBuildQueue:
    def test_add_and_length(self) -> None:
        bq = BuildQueue()
        bq.add("infantry", {ResourceType.CREDITS: 50}, 5.0)
        assert len(bq) == 1

    def test_tick_advances_first_item(self) -> None:
        bq = BuildQueue()
        bq.add("infantry", {}, 5.0)
        bq.add("scout", {}, 3.0)
        bq.tick(3.0)
        # Only first item advances
        assert bq.queue[0]["progress"] == 3.0
        assert bq.queue[1]["progress"] == 0.0

    def test_pop_completed(self) -> None:
        bq = BuildQueue()
        bq.add("infantry", {}, 5.0)
        bq.tick(6.0)  # exceeds build time
        completed = bq.pop_completed()
        assert completed == ["infantry"]
        assert len(bq) == 0

    def test_pop_completed_nothing_done(self) -> None:
        bq = BuildQueue()
        bq.add("infantry", {}, 5.0)
        bq.tick(2.0)
        assert bq.pop_completed() == []
        assert len(bq) == 1

    def test_peek(self) -> None:
        bq = BuildQueue()
        assert bq.peek() is None
        bq.add("infantry", {}, 5.0)
        item = bq.peek()
        assert item is not None
        assert item["unit_template"] == "infantry"

    def test_cancel(self) -> None:
        bq = BuildQueue()
        bq.add("infantry", {}, 5.0)
        bq.add("scout", {}, 3.0)
        cancelled = bq.cancel(0)
        assert cancelled is not None
        assert cancelled["unit_template"] == "infantry"
        assert len(bq) == 1

    def test_cancel_out_of_range(self) -> None:
        bq = BuildQueue()
        assert bq.cancel(0) is None
        bq.add("infantry", {}, 5.0)
        assert bq.cancel(5) is None

    def test_to_list_shows_progress(self) -> None:
        bq = BuildQueue()
        bq.add("infantry", {}, 10.0)
        bq.tick(5.0)
        items = bq.to_list()
        assert len(items) == 1
        assert items[0]["progress_pct"] == 0.5
        assert items[0]["remaining_s"] == 5.0

    def test_empty_queue_tick_no_crash(self) -> None:
        bq = BuildQueue()
        bq.tick(1.0)  # should not raise


# ---------------------------------------------------------------------------
# TechTree
# ---------------------------------------------------------------------------

class TestTechTree:
    def test_available_no_prerequisites(self) -> None:
        tree = TechTree()
        tree.add_tech("basic", {ResourceType.CREDITS: 100})
        assert tree.is_available("basic") is True

    def test_not_available_missing_prerequisite(self) -> None:
        tree = TechTree()
        tree.add_tech("basic", {ResourceType.CREDITS: 100})
        tree.add_tech("advanced", {ResourceType.CREDITS: 200},
                      prerequisites=["basic"])
        assert tree.is_available("advanced") is False

    def test_available_after_prerequisite_researched(self) -> None:
        tree = TechTree()
        tree.add_tech("basic", {ResourceType.CREDITS: 100})
        tree.add_tech("advanced", {ResourceType.CREDITS: 200},
                      prerequisites=["basic"])
        pool = ResourcePool(resources={ResourceType.CREDITS: 500})
        tree.research("basic", pool)
        assert tree.is_available("advanced") is True

    def test_research_spends_resources(self) -> None:
        tree = TechTree()
        tree.add_tech("basic", {ResourceType.CREDITS: 100})
        pool = ResourcePool(resources={ResourceType.CREDITS: 150})
        assert tree.research("basic", pool) is True
        assert pool.get(ResourceType.CREDITS) == 50
        assert "basic" in tree.researched

    def test_research_fails_if_cant_afford(self) -> None:
        tree = TechTree()
        tree.add_tech("basic", {ResourceType.CREDITS: 100})
        pool = ResourcePool(resources={ResourceType.CREDITS: 50})
        assert tree.research("basic", pool) is False
        assert "basic" not in tree.researched
        assert pool.get(ResourceType.CREDITS) == 50

    def test_research_already_researched(self) -> None:
        tree = TechTree()
        tree.add_tech("basic", {ResourceType.CREDITS: 100})
        pool = ResourcePool(resources={ResourceType.CREDITS: 500})
        tree.research("basic", pool)
        assert tree.research("basic", pool) is False  # already done

    def test_get_unlocked_units(self) -> None:
        tree = TechTree()
        tree.add_tech("basic", {ResourceType.CREDITS: 100},
                      unlocks=["infantry", "scout"])
        pool = ResourcePool(resources={ResourceType.CREDITS: 200})
        tree.research("basic", pool)
        unlocked = tree.get_unlocked_units()
        assert "infantry" in unlocked
        assert "scout" in unlocked

    def test_available_techs(self) -> None:
        tree = TechTree()
        tree.add_tech("t1", {ResourceType.CREDITS: 10})
        tree.add_tech("t2", {ResourceType.CREDITS: 10}, prerequisites=["t1"])
        avail = tree.available_techs()
        assert "t1" in avail
        assert "t2" not in avail

    def test_to_dict(self) -> None:
        tree = TechTree()
        tree.add_tech("basic", {ResourceType.CREDITS: 100}, unlocks=["infantry"])
        d = tree.to_dict()
        assert "basic" in d
        assert d["basic"]["available"] is True
        assert d["basic"]["researched"] is False

    def test_unknown_tech_not_available(self) -> None:
        tree = TechTree()
        assert tree.is_available("nonexistent") is False


# ---------------------------------------------------------------------------
# EconomyEngine
# ---------------------------------------------------------------------------

class TestEconomyEngine:
    def test_setup_faction_with_preset(self) -> None:
        eng = EconomyEngine()
        eng.setup_faction("alpha", ECONOMY_PRESETS["standard"])
        assert "alpha" in eng.pools
        assert "alpha" in eng.build_queues
        assert eng.pools["alpha"].get(ResourceType.CREDITS) == 1000

    def test_setup_faction_no_preset(self) -> None:
        eng = EconomyEngine()
        eng.setup_faction("alpha")
        assert "alpha" in eng.pools
        assert eng.pools["alpha"].get(ResourceType.CREDITS) == 0.0

    def test_purchase_unit_success(self) -> None:
        eng = EconomyEngine()
        eng.setup_faction("alpha", ECONOMY_PRESETS["rich"])
        eng.register_unit_costs(UNIT_COSTS)
        assert eng.purchase_unit("alpha", "infantry") is True
        assert len(eng.build_queues["alpha"]) == 1

    def test_purchase_unit_insufficient_funds(self) -> None:
        eng = EconomyEngine()
        eng.setup_faction("alpha", ECONOMY_PRESETS["insurgent"])
        eng.register_unit_costs(UNIT_COSTS)
        # Drain credits
        eng.pools["alpha"].resources[ResourceType.CREDITS] = 0
        assert eng.purchase_unit("alpha", "tank") is False

    def test_purchase_unknown_template(self) -> None:
        eng = EconomyEngine()
        eng.setup_faction("alpha", ECONOMY_PRESETS["standard"])
        eng.register_unit_costs(UNIT_COSTS)
        assert eng.purchase_unit("alpha", "space_marine") is False

    def test_purchase_unknown_faction(self) -> None:
        eng = EconomyEngine()
        eng.register_unit_costs(UNIT_COSTS)
        assert eng.purchase_unit("nonexistent", "infantry") is False

    def test_tick_completes_build(self) -> None:
        eng = EconomyEngine()
        eng.setup_faction("alpha", ECONOMY_PRESETS["unlimited"])
        eng.register_unit_costs(UNIT_COSTS)
        eng.purchase_unit("alpha", "scout")  # build_time=4.0
        completed = eng.tick(5.0)
        assert "alpha:scout" in completed
        assert "scout" in eng._active_units["alpha"]

    def test_tick_income_and_upkeep(self) -> None:
        eng = EconomyEngine()
        eng.setup_faction("alpha", ECONOMY_PRESETS["standard"])
        eng.register_unit_costs(UNIT_COSTS)
        # Add some active units for upkeep
        eng.add_active_unit("alpha", "infantry")
        eng.add_active_unit("alpha", "infantry")
        credits_before = eng.pools["alpha"].get(ResourceType.CREDITS)
        eng.tick(60.0)  # 1 minute
        credits_after = eng.pools["alpha"].get(ResourceType.CREDITS)
        # Income of 10/s for 60s = 600, upkeep 2/min * 2 units * 1 min = 4
        # So net should be around credits_before + 600 - 4 = credits_before + 596
        assert credits_after > credits_before

    def test_remove_active_unit(self) -> None:
        eng = EconomyEngine()
        eng.setup_faction("alpha")
        eng.add_active_unit("alpha", "infantry")
        eng.add_active_unit("alpha", "infantry")
        assert eng.remove_active_unit("alpha", "infantry") is True
        assert len(eng._active_units["alpha"]) == 1
        assert eng.remove_active_unit("alpha", "tank") is False

    def test_research_tech(self) -> None:
        eng = EconomyEngine()
        eng.setup_faction("alpha", ECONOMY_PRESETS["standard"])
        eng.register_tech_tree("alpha", copy.deepcopy(TECH_TREE))
        assert eng.research_tech("alpha", "basic_training") is True

    def test_research_tech_unknown_faction(self) -> None:
        eng = EconomyEngine()
        assert eng.research_tech("nonexistent", "basic_training") is False

    def test_get_economy_status(self) -> None:
        eng = EconomyEngine()
        eng.setup_faction("alpha", ECONOMY_PRESETS["standard"])
        eng.register_unit_costs(UNIT_COSTS)
        eng.register_tech_tree("alpha", copy.deepcopy(TECH_TREE))
        status = eng.get_economy_status("alpha")
        assert status["faction"] == "alpha"
        assert "resources" in status
        assert "build_queue" in status
        assert "tech_researched" in status
        assert "tech_available" in status

    def test_to_three_js(self) -> None:
        eng = EconomyEngine()
        eng.setup_faction("alpha", ECONOMY_PRESETS["standard"])
        eng.register_unit_costs(UNIT_COSTS)
        data = eng.to_three_js("alpha")
        assert "resource_bars" in data
        assert "build_queue" in data
        assert "upkeep_per_min" in data
        assert data["faction"] == "alpha"

    def test_all_presets_valid(self) -> None:
        """Every preset should be usable with setup_faction."""
        for name, preset in ECONOMY_PRESETS.items():
            eng = EconomyEngine()
            eng.setup_faction(name, preset)
            assert name in eng.pools
            assert eng.pools[name].get(ResourceType.CREDITS) > 0 or name == "insurgent"

    def test_default_tech_tree_has_15_techs(self) -> None:
        assert len(TECH_TREE.techs) >= 15

    def test_unit_costs_cover_all_templates(self) -> None:
        expected = {"infantry", "sniper", "heavy", "medic", "engineer",
                    "scout", "drone", "turret", "apc", "tank",
                    "attack_helicopter", "patrol_boat"}
        assert expected.issubset(set(UNIT_COSTS.keys()))
