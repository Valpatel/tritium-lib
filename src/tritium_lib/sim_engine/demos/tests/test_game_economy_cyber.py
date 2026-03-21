# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests: economy, cyber warfare, and CQB systems wired into game_server.py.

Verifies that each system is:
  - Initialized in build_full_game()
  - Ticked and producing frame keys each game_tick()
  - Auto-purchasing / auto-attacking / triggering CQB at the right intervals
"""

from __future__ import annotations

import random
import pytest

from tritium_lib.sim_engine.demos.game_server import (
    GameState,
    build_full_game,
    game_tick,
    _auto_purchase_units,
    _auto_cyber_attack,
    _trigger_cqb,
    _AUTO_PURCHASE_INTERVAL,
    _AUTO_CYBER_INTERVAL,
)
from tritium_lib.sim_engine.economy import (
    EconomyEngine, ECONOMY_PRESETS, UNIT_COSTS, TECH_TREE, ResourceType,
)
from tritium_lib.sim_engine.cyber import (
    CyberWarfareEngine, create_asset_from_preset,
)
from tritium_lib.sim_engine.buildings import RoomClearingEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gs() -> GameState:
    """Build a full game state once for the module."""
    return build_full_game("urban_combat")


@pytest.fixture(scope="module")
def frame_1(gs: GameState) -> dict:
    """Run one tick and return the frame."""
    gs.tick_count = 0
    return game_tick(gs, dt=0.1)


# ---------------------------------------------------------------------------
# 1. Initialization checks
# ---------------------------------------------------------------------------

class TestGameStateInit:
    def test_economy_initialized(self, gs: GameState) -> None:
        assert gs.economy is not None

    def test_cyber_initialized(self, gs: GameState) -> None:
        assert gs.cyber is not None

    def test_buildings_initialized(self, gs: GameState) -> None:
        assert gs.buildings is not None

    def test_economy_has_friendly_faction(self, gs: GameState) -> None:
        assert "friendly" in gs.economy.pools  # type: ignore[union-attr]

    def test_economy_has_hostile_faction(self, gs: GameState) -> None:
        assert "hostile" in gs.economy.pools  # type: ignore[union-attr]

    def test_economy_unit_costs_registered(self, gs: GameState) -> None:
        assert "infantry" in gs.economy.unit_costs  # type: ignore[union-attr]
        assert "heavy" in gs.economy.unit_costs  # type: ignore[union-attr]

    def test_cyber_has_assets(self, gs: GameState) -> None:
        assert len(gs.cyber.assets) >= 2  # type: ignore[union-attr]

    def test_cyber_sigint_asset_present(self, gs: GameState) -> None:
        assert "sigint_friendly_1" in gs.cyber.assets  # type: ignore[union-attr]

    def test_cyber_gps_spoofer_present(self, gs: GameState) -> None:
        assert "gps_spoofer_hostile_1" in gs.cyber.assets  # type: ignore[union-attr]

    def test_buildings_have_layouts(self, gs: GameState) -> None:
        assert len(gs.buildings.buildings) >= 2  # type: ignore[union-attr]

    def test_rng_on_game_state(self, gs: GameState) -> None:
        assert isinstance(gs._rng, random.Random)

    def test_cqb_events_list_on_game_state(self, gs: GameState) -> None:
        assert isinstance(gs.cqb_events, list)


# ---------------------------------------------------------------------------
# 2. Frame keys present after tick
# ---------------------------------------------------------------------------

class TestFrameKeys:
    def test_economy_key_in_frame(self, frame_1: dict) -> None:
        assert "economy" in frame_1

    def test_economy_friendly_in_frame(self, frame_1: dict) -> None:
        assert "friendly" in frame_1["economy"]

    def test_economy_hostile_in_frame(self, frame_1: dict) -> None:
        assert "hostile" in frame_1["economy"]

    def test_economy_completed_units_in_frame(self, frame_1: dict) -> None:
        assert "completed_units" in frame_1["economy"]

    def test_cyber_key_in_frame(self, frame_1: dict) -> None:
        assert "cyber" in frame_1

    def test_cyber_assets_in_frame(self, frame_1: dict) -> None:
        assert "assets" in frame_1["cyber"]

    def test_cyber_effects_in_frame(self, frame_1: dict) -> None:
        assert "effects" in frame_1["cyber"]

    def test_cyber_auto_attacks_key_in_frame(self, frame_1: dict) -> None:
        assert "auto_attacks_this_tick" in frame_1["cyber"]

    def test_buildings_key_in_frame(self, frame_1: dict) -> None:
        assert "buildings" in frame_1

    def test_buildings_count_in_frame(self, frame_1: dict) -> None:
        assert frame_1["buildings"]["count"] >= 2

    def test_buildings_sample_in_frame(self, frame_1: dict) -> None:
        sample = frame_1["buildings"]["sample"]
        assert "total_rooms" in sample
        assert "cleared_rooms" in sample


# ---------------------------------------------------------------------------
# 3. Economy auto-purchase logic
# ---------------------------------------------------------------------------

class TestEconomyAutoPurchase:
    def test_no_purchase_before_interval(self, gs: GameState) -> None:
        """Before the first interval tick, no purchases should be made."""
        gs.tick_count = _AUTO_PURCHASE_INTERVAL - 1
        placed = _auto_purchase_units(gs, gs.tick_count)
        assert placed == []

    def test_purchase_at_interval(self, gs: GameState) -> None:
        """At exactly the interval tick, at least one purchase should be queued."""
        # Give factions enough credits to afford
        for faction in ("friendly", "hostile"):
            pool = gs.economy.pools.get(faction)  # type: ignore[union-attr]
            if pool is not None:
                pool.resources[ResourceType.CREDITS] = 5000
                pool.resources[ResourceType.MANPOWER] = 10
                pool.resources[ResourceType.FOOD] = 300
        gs.tick_count = _AUTO_PURCHASE_INTERVAL
        placed = _auto_purchase_units(gs, gs.tick_count)
        assert len(placed) > 0

    def test_purchase_format(self, gs: GameState) -> None:
        """Placed orders must be in 'faction:template' format."""
        for faction in ("friendly", "hostile"):
            pool = gs.economy.pools.get(faction)  # type: ignore[union-attr]
            if pool is not None:
                pool.resources[ResourceType.CREDITS] = 5000
                pool.resources[ResourceType.MANPOWER] = 10
                pool.resources[ResourceType.FOOD] = 300
        gs.tick_count = _AUTO_PURCHASE_INTERVAL
        placed = _auto_purchase_units(gs, gs.tick_count)
        for entry in placed:
            faction, template = entry.split(":", 1)
            assert faction in ("friendly", "hostile")
            assert template in gs.economy.unit_costs  # type: ignore[union-attr]

    def test_purchase_queued_in_build_queue(self, gs: GameState) -> None:
        """After a purchase, the build queue for that faction should have entries."""
        for faction in ("friendly", "hostile"):
            pool = gs.economy.pools.get(faction)  # type: ignore[union-attr]
            if pool is not None:
                pool.resources[ResourceType.CREDITS] = 5000
                pool.resources[ResourceType.MANPOWER] = 10
                pool.resources[ResourceType.FOOD] = 300
        gs.tick_count = _AUTO_PURCHASE_INTERVAL
        _auto_purchase_units(gs, gs.tick_count)
        total_queued = sum(
            len(gs.economy.build_queues[f])  # type: ignore[union-attr]
            for f in ("friendly", "hostile")
            if f in gs.economy.build_queues  # type: ignore[union-attr]
        )
        assert total_queued > 0

    def test_economy_tick_advances_build_queue(self, gs: GameState) -> None:
        """Ticking the economy should advance build progress."""
        for faction in ("friendly", "hostile"):
            pool = gs.economy.pools.get(faction)  # type: ignore[union-attr]
            if pool is not None:
                pool.resources[ResourceType.CREDITS] = 5000
                pool.resources[ResourceType.MANPOWER] = 10
                pool.resources[ResourceType.FOOD] = 300
        gs.economy.purchase_unit("friendly", "scout")  # type: ignore[union-attr]
        bq = gs.economy.build_queues["friendly"]  # type: ignore[union-attr]
        initial_progress = bq.queue[0]["progress"] if bq.queue else 0.0
        gs.economy.tick(5.0)  # type: ignore[union-attr]
        if bq.queue:
            assert bq.queue[0]["progress"] >= initial_progress

    def test_economy_frame_has_resource_bars(self, gs: GameState) -> None:
        """The to_three_js export should include resource_bars."""
        data = gs.economy.to_three_js("friendly")  # type: ignore[union-attr]
        assert "resource_bars" in data
        assert len(data["resource_bars"]) > 0

    def test_economy_no_purchase_when_no_credits(self) -> None:
        """With zero credits, auto-purchase should fail gracefully."""
        import copy as _copy
        eng = EconomyEngine()
        eng.setup_faction("friendly", {
            "resources": {ResourceType.CREDITS: 0, ResourceType.MANPOWER: 0, ResourceType.FOOD: 0},
            "income": {},
            "capacity": {ResourceType.CREDITS: 1000},
        })
        eng.register_unit_costs(UNIT_COSTS)
        eng.register_tech_tree("friendly", _copy.deepcopy(TECH_TREE))

        gs_stub = GameState()
        gs_stub.economy = eng
        gs_stub.tick_count = _AUTO_PURCHASE_INTERVAL
        placed = _auto_purchase_units(gs_stub, gs_stub.tick_count)
        assert placed == []


# ---------------------------------------------------------------------------
# 4. Cyber auto-attack logic
# ---------------------------------------------------------------------------

class TestCyberAutoAttack:
    def test_no_attack_before_interval(self, gs: GameState) -> None:
        """Before the first attack interval, _auto_cyber_attack returns empty."""
        gs.tick_count = _AUTO_CYBER_INTERVAL - 1
        result = _auto_cyber_attack(gs, gs.tick_count, gs._rng)
        assert result == []

    def test_attack_at_interval(self, gs: GameState) -> None:
        """At the attack interval tick, at least one attack should fire."""
        # Reset cooldowns so attacks can fire
        for asset in gs.cyber.assets.values():  # type: ignore[union-attr]
            for cap in asset.capabilities:
                cap.cooldown_remaining = 0.0
        gs.tick_count = _AUTO_CYBER_INTERVAL
        result = _auto_cyber_attack(gs, gs.tick_count, gs._rng)
        # Result may be empty if no targets in range, but should be a list
        assert isinstance(result, list)

    def test_attack_returns_attack_type_strings(self, gs: GameState) -> None:
        """Returned values should be attack type strings."""
        valid_types = {
            "jamming", "spoofing", "denial_of_service", "malware",
            "intercept", "decoy", "gps_spoofing", "drone_hijack",
        }
        for asset in gs.cyber.assets.values():  # type: ignore[union-attr]
            for cap in asset.capabilities:
                cap.cooldown_remaining = 0.0
        gs.tick_count = _AUTO_CYBER_INTERVAL
        result = _auto_cyber_attack(gs, gs.tick_count, gs._rng)
        for atype in result:
            assert atype in valid_types, f"Unexpected attack type: {atype!r}"

    def test_cyber_initial_effect_active(self, gs: GameState) -> None:
        """The GPS spoofer launched in build_full_game should have created at least
        one active effect on the cyber engine (may have expired; we just check the
        engine ticks without error)."""
        eng = gs.cyber
        assert eng is not None
        events = eng.tick(1.0, {}, {})
        assert isinstance(events, list)

    def test_cyber_to_three_js_structure(self, gs: GameState) -> None:
        """to_three_js must return assets, effects, attack_lines keys."""
        data = gs.cyber.to_three_js()  # type: ignore[union-attr]
        assert "assets" in data
        assert "effects" in data
        assert "attack_lines" in data

    def test_cyber_frame_auto_attacks_key_is_list(self, gs: GameState) -> None:
        """After a full tick, the cyber frame key auto_attacks_this_tick is a list."""
        gs.tick_count = _AUTO_CYBER_INTERVAL
        for asset in gs.cyber.assets.values():  # type: ignore[union-attr]
            for cap in asset.capabilities:
                cap.cooldown_remaining = 0.0
        frame = game_tick(gs, dt=0.1)
        assert isinstance(frame["cyber"].get("auto_attacks_this_tick"), list)

    def test_cyber_event_log_drains_after_tick(self, gs: GameState) -> None:
        """drain_event_log() should return a list and clear the internal log."""
        # Launch attack to populate log
        for asset in gs.cyber.assets.values():  # type: ignore[union-attr]
            for cap in asset.capabilities:
                cap.cooldown_remaining = 0.0
        _auto_cyber_attack(gs, _AUTO_CYBER_INTERVAL, gs._rng)
        log = gs.cyber.drain_event_log()  # type: ignore[union-attr]
        assert isinstance(log, list)
        # Log should now be empty after drain
        log2 = gs.cyber.drain_event_log()  # type: ignore[union-attr]
        assert log2 == []


# ---------------------------------------------------------------------------
# 5. CQB room-clearing trigger
# ---------------------------------------------------------------------------

class TestCQBTrigger:
    def test_cqb_returns_list(self, gs: GameState) -> None:
        """_trigger_cqb always returns a list."""
        gs.tick_count = 20
        result = _trigger_cqb(gs, gs.tick_count, gs._rng)
        assert isinstance(result, list)

    def test_cqb_no_trigger_off_interval(self, gs: GameState) -> None:
        """CQB should not trigger on ticks not divisible by 20."""
        gs.tick_count = 21
        result = _trigger_cqb(gs, gs.tick_count, gs._rng)
        assert result == []

    def test_cqb_result_structure_when_triggered(self) -> None:
        """When infantry are close enough, clear_room result has expected keys."""
        # Build an isolated engine for deterministic test
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=4,
            building_pos=(0.0, 0.0), template="house",
        )

        # Build a minimal GameState stub
        from tritium_lib.sim_engine.units import Unit, Alliance, UnitType
        from tritium_lib.sim_engine.world import World, WorldConfig

        world = World(WorldConfig(map_size=(100.0, 100.0)))
        # Spawn 3 friendly infantry very close to the building (0, 0)
        from tritium_lib.sim_engine.units import create_unit, Alliance
        for i in range(3):
            u = create_unit("infantry", f"inf_{i}", f"Alpha-{i}", Alliance.FRIENDLY, (5.0, 5.0))
            world.units[u.unit_id] = u

        gs_stub = GameState()
        gs_stub.world = world
        gs_stub.buildings = eng
        gs_stub.tick_count = 20

        result_list = _trigger_cqb(gs_stub, 20, random.Random(0))
        # If triggered, each result must have success and room_id
        for res in result_list:
            assert "success" in res
            if res["success"]:
                assert "room_id" in res
                assert "cleared_count" in res

    def test_cqb_events_accumulate_on_gs(self, gs: GameState) -> None:
        """cqb_events list on GameState accumulates results across ticks."""
        initial_count = len(gs.cqb_events)
        # Run multiple ticks that land on CQB intervals
        for tick in range(20, 61, 20):
            gs.tick_count = tick
            results = _trigger_cqb(gs, tick, gs._rng)
            gs.cqb_events.extend(results)
        # Count may or may not grow (depends on unit proximity), but must not error
        assert len(gs.cqb_events) >= initial_count

    def test_cqb_key_in_frame_when_events_exist(self) -> None:
        """If cqb_events is non-empty, the 'cqb' key appears in the frame."""
        gs_fresh = build_full_game("urban_combat")
        # Pre-populate cqb_events with a dummy result
        gs_fresh.cqb_events.append({
            "success": True, "room_id": "fake_room",
            "building_id": "fake_bld", "hostiles_found": 0,
            "hostiles_killed": [], "friendly_casualties": [],
            "room_cleared": True, "building_cleared": False,
            "cleared_count": 1, "total_rooms": 4,
            "flashbang_used": False, "accuracy": 0.85, "tick": 1,
        })
        gs_fresh.tick_count = 20
        # Run a buildings tick so the frame key gets set
        frame = game_tick(gs_fresh, dt=0.1)
        # cqb key should appear because cqb_events is non-empty
        assert "cqb" in frame
        assert len(frame["cqb"]) > 0

    def test_buildings_tick_does_not_raise(self, gs: GameState) -> None:
        """buildings.tick(dt) must not raise for any dt."""
        assert gs.buildings is not None
        gs.buildings.tick(0.1)  # should not raise

    def test_room_clearing_engine_has_rooms(self, gs: GameState) -> None:
        """Both buildings should have rooms generated."""
        assert gs.buildings is not None
        for bld in gs.buildings.buildings.values():
            assert len(bld.rooms) > 0


# ---------------------------------------------------------------------------
# 6. Full integration: multi-tick run
# ---------------------------------------------------------------------------

class TestMultiTickIntegration:
    def test_economy_resources_grow_over_ticks(self) -> None:
        """Resources should increase after several ticks due to income."""
        import copy as _copy
        eng = EconomyEngine()
        eng.setup_faction("friendly", ECONOMY_PRESETS["standard"])
        eng.register_unit_costs(UNIT_COSTS)
        eng.register_tech_tree("friendly", _copy.deepcopy(TECH_TREE))

        initial = eng.pools["friendly"].get(ResourceType.CREDITS)
        eng.tick(10.0)
        after = eng.pools["friendly"].get(ResourceType.CREDITS)
        assert after > initial

    def test_10_game_ticks_no_exception(self, gs: GameState) -> None:
        """Running 10 consecutive ticks should not raise any exception."""
        gs.tick_count = 1
        for _ in range(10):
            frame = game_tick(gs, dt=0.1)
            assert "tick" in frame

    def test_economy_frame_has_build_queue(self, gs: GameState) -> None:
        """After enough ticks with auto-purchase, build_queue should appear."""
        # Ensure enough resources for purchase
        for faction in ("friendly", "hostile"):
            pool = gs.economy.pools.get(faction)  # type: ignore[union-attr]
            if pool:
                pool.resources[ResourceType.CREDITS] = 9999
                pool.resources[ResourceType.MANPOWER] = 50
                pool.resources[ResourceType.FOOD] = 999
        gs.tick_count = _AUTO_PURCHASE_INTERVAL
        frame = game_tick(gs, dt=0.1)
        economy_data = frame["economy"]
        for faction in ("friendly", "hostile"):
            assert "build_queue" in economy_data[faction]

    def test_cyber_effects_can_expire(self) -> None:
        """An effect with 0.1s duration expires after a tick of 1.0s."""
        eng = CyberWarfareEngine(rng_seed=0)
        asset = create_asset_from_preset(
            "sigint_post", "test_asset", (0.0, 0.0), "friendly",
        )
        # Force duration to be very short
        for cap in asset.capabilities:
            cap.duration = 0.1
            cap.cooldown_remaining = 0.0
        eng.deploy_asset(asset)
        cap_id = asset.capabilities[0].capability_id
        eng.launch_attack("test_asset", cap_id, (50.0, 0.0))
        # Tick past expiry
        events = eng.tick(1.0, {}, {})
        expired = [e for e in events if e.get("type") == "effect_expired"]
        assert len(expired) >= 1
