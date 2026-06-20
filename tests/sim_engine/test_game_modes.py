# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for game mode variety — battle, civil_unrest, drone_swarm.

Verifies that each game mode type produces different gameplay behavior:
different scoring, different win/loss conditions, different state fields,
and different event publishing.
"""

import time
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

import pytest

from tritium_lib.sim_engine.game.game_mode import (
    GameMode,
    WaveConfig,
    WAVE_CONFIGS,
    InfiniteWaveMode,
    InstigatorDetector,
    _IDENTIFICATION_SCORE,
)


# ---------------------------------------------------------------------------
# Minimal stubs for duck-typed dependencies
# ---------------------------------------------------------------------------


@dataclass
class _FakeTarget:
    target_id: str
    name: str = "Unit"
    alliance: str = "friendly"
    is_combatant: bool = True
    status: str = "active"
    battery: float = 1.0
    health: float = 100.0
    max_health: float = 100.0
    asset_type: str = "person"
    speed: float = 5.0
    position: tuple[float, float] = (0.0, 0.0)


class _FakeEngine:
    """Minimal engine stub for GameMode."""

    def __init__(self):
        self._targets: dict[str, _FakeTarget] = {}
        self._map_bounds = 100.0
        self.hazard_manager = MagicMock()
        self.stats_tracker = MagicMock()

    def get_targets(self):
        return list(self._targets.values())

    def spawn_hostile(self, direction="random"):
        tid = f"hostile_{len(self._targets)}"
        t = _FakeTarget(target_id=tid, alliance="hostile", status="active")
        self._targets[tid] = t
        return t

    def spawn_hostile_typed(self, asset_type="person", speed=1.0, health=100.0,
                           drone_variant=None):
        return self.spawn_hostile()

    def add_target(self, target):
        self._targets[target.target_id] = target

    def set_map_bounds(self, bounds):
        self._map_bounds = bounds

    def add_friendly(self):
        tid = f"friendly_{len(self._targets)}"
        t = _FakeTarget(target_id=tid, alliance="friendly", status="active")
        self._targets[tid] = t
        return t


class _FakeCombat:
    def reset_streaks(self): pass
    def clear(self): pass


class _FakeEventBus:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def publish(self, event_name, data):
        self.events.append((event_name, data))


def _build_game_mode(game_mode_type="battle", infinite=False):
    """Create a GameMode with stub dependencies and friendlies already alive."""
    bus = _FakeEventBus()
    engine = _FakeEngine()
    combat = _FakeCombat()

    # Add 3 friendly combatants so defeat checks don't trigger immediately
    for _ in range(3):
        engine.add_friendly()

    gm = GameMode(event_bus=bus, engine=engine, combat_system=combat, infinite=infinite)
    gm.game_mode_type = game_mode_type

    return gm, bus, engine


# ===================================================================
# Battle mode (default)
# ===================================================================


class TestBattleMode:
    """Default battle mode: wave-based, score by eliminations."""

    def test_default_mode_type_is_battle(self):
        gm, _, _ = _build_game_mode()
        assert gm.game_mode_type == "battle"

    def test_state_starts_at_setup(self):
        gm, _, _ = _build_game_mode()
        assert gm.state == "setup"

    def test_begin_war_transitions_to_countdown(self):
        gm, _, _ = _build_game_mode()
        gm.begin_war()
        assert gm.state == "countdown"

    def test_score_increments_on_elimination(self):
        gm, _, _ = _build_game_mode()
        gm.begin_war()
        gm.state = "active"  # skip countdown
        gm.on_target_eliminated("hostile_0")
        assert gm.score == 100
        assert gm.total_eliminations == 1

    def test_get_state_has_battle_fields(self):
        gm, _, _ = _build_game_mode()
        gm.begin_war()
        state = gm.get_state()
        assert state["game_mode_type"] == "battle"
        assert "score" in state
        assert "wave" in state
        assert "total_eliminations" in state

    def test_get_state_no_civil_unrest_fields(self):
        """Battle mode should NOT have civil_unrest-specific fields."""
        gm, _, _ = _build_game_mode()
        state = gm.get_state()
        assert "de_escalation_score" not in state
        assert "civilian_harm_count" not in state
        assert "infrastructure_health" not in state

    def test_wave_configs_exist(self):
        """Default battle mode should have 10 hardcoded wave configs."""
        assert len(WAVE_CONFIGS) == 10

    def test_waves_increase_in_difficulty(self):
        """Later waves should have more hostiles and/or higher multipliers."""
        first = WAVE_CONFIGS[0]
        last = WAVE_CONFIGS[-1]
        assert last.count > first.count
        assert last.health_mult >= first.health_mult

    def test_spawn_directions_vary(self):
        """Waves should use different spawn directions for variety."""
        directions = {w.spawn_direction for w in WAVE_CONFIGS}
        assert len(directions) >= 2, (
            f"Expected varied spawn directions, got {directions}"
        )

    def test_later_waves_have_mixed_composition(self):
        """Waves 3+ should have mixed unit composition (vehicles, leaders)."""
        mixed = [w for w in WAVE_CONFIGS if w.composition is not None]
        assert len(mixed) >= 5, "Most waves should have mixed compositions"

    def test_unit_types_in_compositions(self):
        """Compositions should include more than just 'person'."""
        all_types = set()
        for w in WAVE_CONFIGS:
            if w.composition:
                for asset_type, count in w.composition:
                    all_types.add(asset_type)
        assert "person" in all_types
        assert "hostile_vehicle" in all_types
        assert "hostile_leader" in all_types


# ===================================================================
# Civil Unrest mode
# ===================================================================


class TestCivilUnrestMode:
    """Civil unrest: de-escalation scoring, civilian harm limits."""

    def test_mode_type(self):
        gm, _, _ = _build_game_mode("civil_unrest")
        assert gm.game_mode_type == "civil_unrest"

    def test_get_state_has_civil_unrest_fields(self):
        """Civil unrest state should include de-escalation and harm fields."""
        gm, _, _ = _build_game_mode("civil_unrest")
        state = gm.get_state()
        assert "de_escalation_score" in state
        assert "civilian_harm_count" in state
        assert "civilian_harm_limit" in state
        assert "weighted_total_score" in state

    def test_civilian_harm_increases_count(self):
        gm, _, _ = _build_game_mode("civil_unrest")
        gm.begin_war()
        gm.state = "active"
        gm.on_civilian_harmed()
        assert gm.civilian_harm_count == 1

    def test_civilian_harm_reduces_de_escalation_score(self):
        gm, _, _ = _build_game_mode("civil_unrest")
        gm.begin_war()
        gm.state = "active"
        gm.de_escalation_score = 1000
        gm.on_civilian_harmed()
        assert gm.de_escalation_score == 500  # -500

    def test_excessive_force_causes_defeat(self):
        """Harming too many civilians should trigger defeat."""
        gm, bus, _ = _build_game_mode("civil_unrest")
        gm.begin_war()
        gm.state = "active"
        gm.civilian_harm_limit = 3  # low limit for testing

        for _ in range(3):
            gm.on_civilian_harmed()

        assert gm.state == "defeat"
        # Check game_over event was published
        game_over_events = [e for name, e in bus.events if name == "game_over"]
        assert len(game_over_events) >= 1
        assert game_over_events[0]["reason"] == "excessive_force"

    def test_weighted_score_formula(self):
        """Weighted score should be 30% combat + 70% de-escalation."""
        gm, _, _ = _build_game_mode("civil_unrest")
        gm.score = 1000
        gm.de_escalation_score = 2000
        state = gm.get_state()
        expected = int(1000 * 0.3 + 2000 * 0.7)
        assert state["weighted_total_score"] == expected

    def test_game_over_data_includes_civil_unrest_fields(self):
        """Game over data in civil_unrest should include mode-specific fields."""
        gm, bus, _ = _build_game_mode("civil_unrest")
        gm.begin_war()
        gm.state = "active"
        gm.de_escalation_score = 500
        gm.civilian_harm_limit = 1  # trigger defeat on first harm
        gm.on_civilian_harmed()

        game_over_events = [e for name, e in bus.events if name == "game_over"]
        assert len(game_over_events) >= 1
        data = game_over_events[0]
        assert "de_escalation_score" in data
        assert "civilian_harm_count" in data
        assert "weighted_total_score" in data


class TestCivilUnrestDeEscalationVictory:
    """Order-restored WIN: civil_unrest is won by de-escalation, not only by
    grinding every attrition wave (roadmap #1 — keeps the mode completable)."""

    def test_target_defaults_to_disabled(self):
        gm, _, _ = _build_game_mode("civil_unrest")
        assert gm.de_escalation_target == 0

    def _order_restored(self, bus):
        return [e for name, e in bus.events
                if name == "game_over" and e.get("reason") == "order_restored"]

    def test_no_victory_when_target_disabled(self):
        """target == 0 preserves the legacy all_waves_cleared-only behavior."""
        gm, bus, _ = _build_game_mode("civil_unrest")
        gm.begin_war()
        gm.state = "active"
        gm.de_escalation_score = 99999
        gm.tick(0.1)
        assert gm.state != "victory"
        assert not self._order_restored(bus)

    def test_victory_when_score_reaches_target(self):
        gm, bus, _ = _build_game_mode("civil_unrest")
        gm.begin_war()
        gm.state = "active"
        gm.de_escalation_target = 500
        gm.de_escalation_score = 500
        gm.tick(0.1)
        assert gm.state == "victory"
        game_over = [e for name, e in bus.events if name == "game_over"]
        assert len(game_over) == 1
        assert game_over[0]["result"] == "victory"
        assert game_over[0]["reason"] == "order_restored"

    def test_no_victory_below_target(self):
        gm, bus, _ = _build_game_mode("civil_unrest")
        gm.begin_war()
        gm.state = "active"
        gm.de_escalation_target = 500
        gm.de_escalation_score = 499
        gm.tick(0.1)
        assert gm.state != "victory"
        assert not self._order_restored(bus)

    def test_target_only_applies_to_civil_unrest(self):
        """A battle game with a stray target must not order-restore."""
        gm, bus, _ = _build_game_mode("battle")
        gm.begin_war()
        gm.state = "active"
        gm.de_escalation_target = 100
        gm.de_escalation_score = 999
        gm.tick(0.1)
        assert gm.state != "victory"
        assert not self._order_restored(bus)

    def test_target_in_get_state_and_game_over(self):
        gm, _, _ = _build_game_mode("civil_unrest")
        gm.de_escalation_target = 600
        assert gm.get_state()["de_escalation_target"] == 600
        data = gm._build_game_over_data("victory", reason="order_restored")
        assert data["de_escalation_target"] == 600

    def test_reset_clears_target(self):
        gm, _, _ = _build_game_mode("civil_unrest")
        gm.de_escalation_target = 600
        gm.reset()
        assert gm.de_escalation_target == 0

    def test_load_scenario_applies_target_from_mode_config(self):
        """The scenario->game_mode seam: mode_config carries the target."""
        gm, _, _ = _build_game_mode("civil_unrest")

        class _StubScenario:
            waves = []
            defenders = []
            map_bounds = 100.0
            mode_config = {"de_escalation_target": 750, "civilian_harm_limit": 3}

        gm.load_scenario(_StubScenario())
        assert gm.de_escalation_target == 750
        assert gm.civilian_harm_limit == 3


class TestInstigatorDetain:
    """InstigatorDetector._identify NON-LETHALLY detains the ringleader and
    drives the de-escalation victory loop. Identifying a ringleader must:
    award de-escalation score, convert it to a neutral non-combatant (so it
    leaves the hostile headcount and stops recruiting), and publish both the
    ``instigator_identified`` and ``de_escalation`` events. Without the detain
    + score, ``order_restored`` victory is mechanically unreachable."""

    def _make_instigator(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            target_id="rioter_7",
            alliance="hostile",
            is_combatant=True,
            crowd_role="instigator",
            identified=False,
            position=(10.0, 10.0),
            weapon_range=8.0,
            weapon_damage=5.0,
            weapon_cooldown=1.5,
        )

    def _make_scout(self):
        from types import SimpleNamespace
        return SimpleNamespace(target_id="scout_1", position=(10.0, 10.0))

    def test_identify_detains_non_lethally(self):
        bus = _FakeEventBus()
        det = InstigatorDetector(event_bus=bus)
        inst = self._make_instigator()
        det._identify(inst, self._make_scout())

        assert inst.identified is True
        # Non-lethal: neutralized, disarmed, dropped from hostile headcount.
        assert inst.alliance == "neutral"
        assert inst.is_combatant is False
        assert inst.crowd_role == "calmed"
        assert inst.weapon_range == 0.0
        assert inst.weapon_damage == 0.0
        assert inst.weapon_cooldown == 0.0

    def test_identify_publishes_both_events(self):
        bus = _FakeEventBus()
        det = InstigatorDetector(event_bus=bus)
        det._identify(self._make_instigator(), self._make_scout())

        names = [n for n, _ in bus.events]
        assert "instigator_identified" in names
        assert "de_escalation" in names

    def test_identify_awards_de_escalation_score(self):
        bus = _FakeEventBus()
        gm, _, _ = _build_game_mode("civil_unrest")
        gm.de_escalation_score = 0
        det = InstigatorDetector(event_bus=bus, game_mode=gm)
        det._identify(self._make_instigator(), self._make_scout())
        assert gm.de_escalation_score == _IDENTIFICATION_SCORE


class TestCalmingCascade:
    """Detaining a ringleader COOLS the local crowd (ESIM doctrine: crowds are
    not uniformly violent; removing the instigator de-escalates the cluster
    around it). When ``_calm_nearby`` runs on a detained ringleader it must:
    revert nearby rioters to non-combatant civilians, and reset nearby ACTIVE
    (still-hidden-identity) instigators back to their hidden/passive state so
    they stop throwing objects -- WITHOUT identifying them or awarding score
    (the win still requires genuine identifications, the cascade only lowers
    attack pressure so defenders survive to finish the loop). Out-of-radius
    units and already-calmed units are untouched."""

    def _ns(self, **kw):
        from types import SimpleNamespace
        base = dict(
            alliance="hostile", is_combatant=True, status="active",
            identified=False, instigator_state="active", instigator_timer=4.0,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def _detained(self):
        return self._ns(target_id="ring_0", position=(0.0, 0.0),
                        crowd_role="calmed", alliance="neutral",
                        is_combatant=False, identified=True)

    def test_nearby_rioter_reverts_to_civilian(self):
        det = InstigatorDetector(event_bus=_FakeEventBus())
        ring = self._detained()
        rioter = self._ns(target_id="rt_1", position=(5.0, 0.0),
                          crowd_role="rioter")
        n = det._calm_nearby(ring, {"ring_0": ring, "rt_1": rioter})
        assert rioter.crowd_role == "civilian"
        assert rioter.is_combatant is False
        assert n == 1

    def test_nearby_active_instigator_resets_to_hidden(self):
        det = InstigatorDetector(event_bus=_FakeEventBus())
        ring = self._detained()
        other = self._ns(target_id="ig_2", position=(8.0, 0.0),
                         crowd_role="instigator")
        det._calm_nearby(ring, {"ring_0": ring, "ig_2": other})
        # Pacified but NOT identified -- still an instigator, just cooled off.
        assert other.instigator_state == "hidden"
        assert other.instigator_timer == 0.0
        assert other.identified is False
        assert other.crowd_role == "instigator"

    def test_far_units_untouched(self):
        det = InstigatorDetector(event_bus=_FakeEventBus())
        ring = self._detained()
        far = self._ns(target_id="rt_far", position=(500.0, 0.0),
                       crowd_role="rioter")
        n = det._calm_nearby(ring, {"ring_0": ring, "rt_far": far})
        assert far.crowd_role == "rioter"
        assert far.is_combatant is True
        assert n == 0

    def test_cascade_does_not_award_score(self):
        """The cascade lowers attack pressure but must NOT inflate the metric:
        only genuine identifications add de_escalation_score."""
        gm, _, _ = _build_game_mode("civil_unrest")
        gm.de_escalation_score = 0
        det = InstigatorDetector(event_bus=_FakeEventBus(), game_mode=gm)
        ring = self._detained()
        rioter = self._ns(target_id="rt_1", position=(3.0, 0.0),
                          crowd_role="rioter")
        det._calm_nearby(ring, {"ring_0": ring, "rt_1": rioter})
        assert gm.de_escalation_score == 0

    def test_identify_triggers_cascade_in_tick(self):
        """End-to-end via tick(): a scout that identifies a ringleader also
        calms a rioter standing next to it."""
        det = InstigatorDetector(event_bus=_FakeEventBus(), detection_time=0.1)
        scout = self._ns(target_id="scout_1", alliance="friendly",
                         asset_type="scout_drone", crowd_role=None,
                         position=(20.0, 20.0))
        ring = self._ns(target_id="ring_0", crowd_role="instigator",
                        position=(20.0, 20.0), weapon_range=8.0,
                        weapon_damage=5.0, weapon_cooldown=1.5)
        rioter = self._ns(target_id="rt_1", crowd_role="rioter",
                          position=(22.0, 20.0))
        targets = {"scout_1": scout, "ring_0": ring, "rt_1": rioter}
        det.tick(0.2, targets, "civil_unrest")
        assert ring.identified is True
        assert rioter.crowd_role == "civilian"


# ===================================================================
# Drone Swarm mode
# ===================================================================


class TestDroneSwarmMode:
    """Drone swarm: infrastructure defense, health-based defeat."""

    def test_mode_type(self):
        gm, _, _ = _build_game_mode("drone_swarm")
        assert gm.game_mode_type == "drone_swarm"

    def test_get_state_has_infrastructure_fields(self):
        """Drone swarm state should include infrastructure health."""
        gm, _, _ = _build_game_mode("drone_swarm")
        gm.infrastructure_health = 500.0
        gm.infrastructure_max = 1000.0
        state = gm.get_state()
        assert "infrastructure_health" in state
        assert "infrastructure_max" in state

    def test_infrastructure_damage_reduces_health(self):
        gm, _, _ = _build_game_mode("drone_swarm")
        gm.begin_war()
        gm.state = "active"
        gm.infrastructure_health = 100.0
        gm.infrastructure_max = 100.0
        gm.on_infrastructure_damaged(30.0)
        assert gm.infrastructure_health == pytest.approx(70.0)

    def test_infrastructure_destruction_causes_defeat(self):
        """Infrastructure reaching 0 health should trigger defeat."""
        gm, bus, _ = _build_game_mode("drone_swarm")
        gm.begin_war()
        gm.state = "active"
        gm.infrastructure_health = 50.0
        gm.infrastructure_max = 100.0

        gm.on_infrastructure_damaged(60.0)  # drops to 0

        assert gm.state == "defeat"
        game_over_events = [e for name, e in bus.events if name == "game_over"]
        assert len(game_over_events) >= 1
        assert game_over_events[0]["reason"] == "infrastructure_destroyed"

    def test_infrastructure_health_clamped_at_zero(self):
        gm, _, _ = _build_game_mode("drone_swarm")
        gm.begin_war()
        gm.state = "active"
        gm.infrastructure_health = 10.0
        gm.on_infrastructure_damaged(100.0)
        assert gm.infrastructure_health == 0.0

    def test_game_over_data_includes_infrastructure(self):
        gm, bus, _ = _build_game_mode("drone_swarm")
        gm.begin_war()
        gm.state = "active"
        gm.infrastructure_health = 5.0
        gm.on_infrastructure_damaged(10.0)

        game_over_events = [e for name, e in bus.events if name == "game_over"]
        assert len(game_over_events) >= 1
        data = game_over_events[0]
        assert "infrastructure_health" in data
        assert "infrastructure_max" in data


# ===================================================================
# Infinite mode
# ===================================================================


class TestInfiniteMode:
    """Verify infinite mode generates procedural waves beyond wave 10."""

    def test_infinite_flag_set(self):
        gm, _, _ = _build_game_mode(infinite=True)
        assert gm.infinite is True

    def test_infinite_wave_generator_exists(self):
        gm, _, _ = _build_game_mode(infinite=True)
        assert gm._infinite_wave_mode is not None

    def test_wave_11_generated(self):
        """Beyond wave 10, infinite mode should generate wave configs."""
        iwm = InfiniteWaveMode()
        config = iwm.get_wave_config(11)
        assert config is not None
        assert config.count > 0
        assert config.speed_mult > 1.0
        assert config.name  # should have a name

    def test_wave_count_grows(self):
        """Higher wave numbers should have more hostiles."""
        iwm = InfiniteWaveMode()
        c11 = iwm.get_wave_config(11)
        c30 = iwm.get_wave_config(30)
        assert c30.count > c11.count

    def test_wave_21_has_boss(self):
        """Wave 21 is the first boss wave (>20, every 5 after that)."""
        iwm = InfiniteWaveMode()
        c21 = iwm.get_wave_config(21)
        assert c21.has_boss is True
        assert "BOSS" in c21.name

    def test_wave_10_has_elites(self):
        """Wave 10+ should have elite enemies."""
        iwm = InfiniteWaveMode()
        c15 = iwm.get_wave_config(15)
        assert c15.has_elites is True

    def test_infinite_get_state_total_waves_minus_one(self):
        """In infinite mode, total_waves should be -1."""
        gm, _, _ = _build_game_mode(infinite=True)
        state = gm.get_state()
        assert state["total_waves"] == -1

    def test_score_multiplier_grows(self):
        """Higher waves should have higher score multipliers."""
        iwm = InfiniteWaveMode()
        c11 = iwm.get_wave_config(11)
        c50 = iwm.get_wave_config(50)
        assert c50.score_mult >= c11.score_mult


# ===================================================================
# Cross-mode comparison
# ===================================================================


class TestCrossModeComparison:
    """Verify modes produce meaningfully different gameplay."""

    def test_each_mode_has_unique_state_fields(self):
        """Each mode should export different state keys."""
        states = {}
        for mode in ("battle", "civil_unrest", "drone_swarm"):
            gm, _, _ = _build_game_mode(mode)
            states[mode] = set(gm.get_state().keys())

        # Civil unrest has de_escalation_score, battle doesn't
        assert "de_escalation_score" in states["civil_unrest"]
        assert "de_escalation_score" not in states["battle"]
        assert "de_escalation_score" not in states["drone_swarm"]

        # Drone swarm has infrastructure, battle doesn't
        gm_drone, _, _ = _build_game_mode("drone_swarm")
        gm_drone.infrastructure_health = 100.0
        drone_state = gm_drone.get_state()
        assert "infrastructure_health" in drone_state

    def test_modes_share_common_fields(self):
        """All modes should have common fields like score, wave, state."""
        common = {"state", "wave", "score", "game_mode_type"}
        for mode in ("battle", "civil_unrest", "drone_swarm"):
            gm, _, _ = _build_game_mode(mode)
            state_keys = set(gm.get_state().keys())
            assert common.issubset(state_keys), (
                f"Mode '{mode}' missing common fields: {common - state_keys}"
            )

    def test_reset_clears_mode_to_battle(self):
        """Resetting should return mode_type to 'battle'."""
        for mode in ("civil_unrest", "drone_swarm"):
            gm, _, _ = _build_game_mode(mode)
            gm.begin_war()
            gm.reset()
            assert gm.game_mode_type == "battle"
            assert gm.state == "setup"

    def test_civilian_harm_only_matters_in_civil_unrest(self):
        """Civilian harm events should be ignored in battle mode."""
        gm, bus, _ = _build_game_mode("battle")
        gm.begin_war()
        gm.state = "active"
        gm.on_civilian_harmed()
        # In battle mode, civilian_harm_count still increments (field exists)
        # but it should NOT cause defeat
        assert gm.state == "active"

    def test_infrastructure_damage_only_matters_in_drone_swarm(self):
        """Infrastructure damage should not cause defeat in battle mode."""
        gm, _, _ = _build_game_mode("battle")
        gm.begin_war()
        gm.state = "active"
        gm.infrastructure_health = 50.0
        gm.on_infrastructure_damaged(100.0)
        # In battle mode this still reduces the counter but doesn't end game
        assert gm.state == "active"


# ===================================================================
# Difficulty scaler
# ===================================================================


class TestDifficultyScaler:
    """Verify the adaptive difficulty system works within game modes."""

    def test_difficulty_initialized(self):
        gm, _, _ = _build_game_mode()
        assert gm.difficulty is not None

    def test_difficulty_multiplier_starts_at_one(self):
        gm, _, _ = _build_game_mode()
        assert gm.difficulty.get_multiplier() == pytest.approx(1.0)

    def test_difficulty_in_state(self):
        """Game state should include difficulty_multiplier."""
        gm, _, _ = _build_game_mode()
        state = gm.get_state()
        assert "difficulty_multiplier" in state

    def test_reset_resets_difficulty(self):
        gm, _, _ = _build_game_mode()
        gm.begin_war()
        gm.reset()
        assert gm.difficulty.get_multiplier() == pytest.approx(1.0)


# ===================================================================
# Wave/score accounting — Village Idiot r5 number-soup fixes
# ===================================================================


class TestWaveDisplayClamp:
    """get_state() is the display/API serialization boundary.

    The engine legitimately increments self.wave to total+1 on the final
    wave_complete tick (that is how victory is detected), but the
    serialized state must never report wave > total_waves to consumers
    (VI r5: top HUD showed 'WAVE 8' for a 7-wave scenario).
    """

    def test_get_state_wave_clamped_at_victory(self):
        gm, _, _ = _build_game_mode()
        gm.state = "victory"
        gm.wave = len(WAVE_CONFIGS) + 1  # engine-internal overflow
        state = gm.get_state()
        assert state["wave"] == len(WAVE_CONFIGS)
        assert state["total_waves"] == len(WAVE_CONFIGS)

    def test_get_state_wave_unclamped_during_normal_play(self):
        gm, _, _ = _build_game_mode()
        gm.state = "active"
        gm.wave = 3
        state = gm.get_state()
        assert state["wave"] == 3

    def test_get_state_wave_not_clamped_in_infinite_mode(self):
        gm, _, _ = _build_game_mode(infinite=True)
        gm.state = "active"
        gm.wave = len(WAVE_CONFIGS) + 5
        state = gm.get_state()
        assert state["wave"] == len(WAVE_CONFIGS) + 5
        assert state["total_waves"] == -1

    def test_get_state_wave_clamped_with_scenario_waves(self):
        gm, _, _ = _build_game_mode()
        gm._scenario_waves = [None] * 7  # 7-wave scenario
        gm.state = "victory"
        gm.wave = 8
        state = gm.get_state()
        assert state["wave"] == 7
        assert state["total_waves"] == 7


class TestEliminationAccounting:
    """Each eliminated target counts exactly once, and only hostiles score.

    VI r5: stalemate force-eliminations incremented the counters
    directly AND re-counted when the published target_eliminated event
    echoed back through the engine listener — every timeout kill counted
    twice and earned +100 unearned score.  Friendly/neutral deaths also
    scored +100 because on_target_eliminated never checked alliance.
    """

    def test_same_target_never_counted_twice(self):
        gm, _, _ = _build_game_mode()
        gm.begin_war()
        gm.state = "active"
        gm.on_target_eliminated("h_dup")
        gm.on_target_eliminated("h_dup")
        assert gm.total_eliminations == 1
        assert gm.score == 100

    def test_force_eliminate_then_event_echo_counts_once(self):
        gm, bus, engine = _build_game_mode()
        gm.begin_war()
        gm.state = "active"
        hostile = engine.spawn_hostile()
        gm._wave_hostile_ids.add(hostile.target_id)
        score_before = gm.score

        gm._force_eliminate_wave_hostiles()
        assert gm.total_eliminations == 1
        assert gm.wave_eliminations == 1

        # Echo: the engine's combat listener forwards the published
        # target_eliminated event back into on_target_eliminated.
        gm.on_target_eliminated(hostile.target_id)
        assert gm.total_eliminations == 1, "echo must not double count"
        assert gm.wave_eliminations == 1, "echo must not double count"
        assert gm.score == score_before, "timeout kills earn no score"

    def test_friendly_death_does_not_score(self):
        gm, _, engine = _build_game_mode()
        gm.begin_war()
        gm.state = "active"
        friendly = engine.get_targets()[0]
        assert friendly.alliance == "friendly"
        gm.on_target_eliminated(friendly.target_id)
        assert gm.total_eliminations == 0
        assert gm.score == 0

    def test_neutral_death_does_not_score(self):
        gm, _, engine = _build_game_mode()
        gm.begin_war()
        gm.state = "active"
        neutral = _FakeTarget(target_id="ped_1", alliance="neutral")
        engine.add_target(neutral)
        gm.on_target_eliminated("ped_1")
        assert gm.total_eliminations == 0
        assert gm.score == 0

    def test_unknown_target_still_scores(self):
        # Pinned behavior: ids the engine no longer knows (pruned) get
        # the benefit of the doubt — they were hostiles.
        gm, _, _ = _build_game_mode()
        gm.begin_war()
        gm.state = "active"
        gm.on_target_eliminated("already_pruned")
        assert gm.total_eliminations == 1
        assert gm.score == 100

    def test_begin_war_clears_dedup_set(self):
        gm, _, _ = _build_game_mode()
        gm.begin_war()
        gm.state = "active"
        gm.on_target_eliminated("h1")
        gm.reset()
        gm.begin_war()
        gm.state = "active"
        gm.on_target_eliminated("h1")
        assert gm.total_eliminations == 1
        assert gm.score == 100


class TestGameOverPayloadClamp:
    def test_game_over_wave_clamped(self):
        gm, bus, _ = _build_game_mode()
        gm.wave = len(WAVE_CONFIGS) + 1
        data = gm._build_game_over_data("victory", waves_completed=len(WAVE_CONFIGS))
        assert data["wave"] == len(WAVE_CONFIGS)
        assert data["total_waves"] == len(WAVE_CONFIGS)

    def test_game_over_wave_raw_in_infinite(self):
        gm, _, _ = _build_game_mode(infinite=True)
        gm.wave = 23
        data = gm._build_game_over_data("defeat")
        assert data["wave"] == 23
        assert data["total_waves"] == -1


class TestLeakStakes:
    """Leaking hostiles past the defense must have stakes, not be rewarded like
    a kill (FEATURE-AUDIT 2026-06-14).

    Before: the full wave*200 bonus was awarded on wave-complete regardless of
    how many hostiles escaped; escapes only fed adaptive difficulty and were
    invisible to the operator.  Now the wave bonus scales by the fraction
    DEFEATED, and leaked counts are tracked + surfaced.
    """

    def _complete_wave(self, spawned, eliminated, wave=1):
        gm, bus, engine = _build_game_mode()
        gm.wave = wave
        gm._wave_hostile_ids = {f"h{i}" for i in range(spawned)}
        gm.wave_eliminations = eliminated
        # Leaks are counted directly from escaped status now; the non-eliminated
        # hostiles are the ones that leaked.
        gm._wave_escaped_ids = {f"h{i}" for i in range(eliminated, spawned)}
        gm._wave_start_time = gm._sim_time  # elapsed 0 -> time_bonus 50
        gm.score = 0
        gm._on_wave_complete()
        return gm, bus

    def test_clean_wave_gets_full_bonus(self):
        gm, _ = self._complete_wave(spawned=4, eliminated=4)
        assert gm.wave_leaked == 0
        assert gm.score == 1 * 200 + 50  # full wave bonus + time bonus (unchanged)

    def test_half_leaked_halves_wave_bonus(self):
        gm, _ = self._complete_wave(spawned=4, eliminated=2)
        assert gm.wave_leaked == 2
        assert gm.total_leaked == 2
        assert gm.score == int(1 * 200 * 0.5) + 50  # 100 + 50

    def test_all_leaked_forfeits_wave_bonus(self):
        gm, _ = self._complete_wave(spawned=4, eliminated=0)
        assert gm.wave_leaked == 4
        assert gm.score == 0 + 50  # only the time bonus; no wave bonus earned

    def test_leaked_counts_surface_in_state(self):
        gm, _ = self._complete_wave(spawned=3, eliminated=1)
        st = gm.get_state()
        assert st["wave_leaked"] == 2
        assert st["total_leaked"] == 2

    def test_wave_complete_event_reports_escaped(self):
        gm, bus = self._complete_wave(spawned=5, eliminated=3)
        wc = [d for (name, d) in bus.events if name == "wave_complete"]
        assert wc, "a wave_complete event should be published"
        assert wc[-1]["escaped"] == 2
        assert wc[-1]["hostiles_spawned"] == 5

    def test_reset_clears_leaked(self):
        gm, _ = self._complete_wave(spawned=4, eliminated=1)
        assert gm.total_leaked == 3
        gm.reset()
        assert gm.total_leaked == 0
        assert gm.wave_leaked == 0


class TestLeakCountRobustness:
    """Leaks are counted from escaped STATUS, not spawned-minus-eliminations, so
    the count stays correct even when the elimination counter is 0 (e.g. a
    headless/replay path with no event-bus wiring).  Without this, every
    defeated hostile is miscounted as a leak (FEATURE-AUDIT 2026-06-14).
    """

    def test_track_escapes_reads_status(self):
        gm, bus, engine = _build_game_mode()
        h = [engine.spawn_hostile() for _ in range(3)]
        for t in h:
            gm._wave_hostile_ids.add(t.target_id)
        h[0].status = "escaped"
        gm._track_escapes()
        assert gm._wave_escaped_ids == {h[0].target_id}
        # idempotent + dedup
        gm._track_escapes()
        assert len(gm._wave_escaped_ids) == 1

    def test_leak_count_independent_of_elimination_counter(self):
        gm, bus, engine = _build_game_mode()
        gm.wave = 1
        h = [engine.spawn_hostile() for _ in range(3)]
        for t in h:
            gm._wave_hostile_ids.add(t.target_id)
        # One escapes; the other two were defeated but the elimination event
        # never fired (wave_eliminations stays 0, as in an unwired harness).
        h[0].status = "escaped"
        gm.wave_eliminations = 0
        gm._wave_start_time = gm._sim_time
        gm.score = 0
        gm._on_wave_complete()
        assert gm.wave_leaked == 1, "only the actually-escaped hostile is a leak"
        # defeat fraction 2/3 -> wave bonus int(200*2/3)=133, + time 50
        assert gm.score == int(1 * 200 * (2 / 3)) + 50


class TestLowBatteryHostilesCountAlive:
    """A recharging (low_battery) hostile is still on the map and a threat, so it
    must count as alive -- otherwise a wave completes / mis-accounts while it
    recharges (FEATURE-AUDIT 2026-06-14, self-audit #11)."""

    def test_low_battery_hostile_counts_alive_and_in_hp(self):
        gm, bus, engine = _build_game_mode()
        ids = []
        for _ in range(3):
            h = engine.spawn_hostile()
            ids.append(h.target_id)
            gm._wave_hostile_ids.add(h.target_id)
        engine._targets[ids[0]].status = "low_battery"  # one recharging
        assert gm._count_wave_hostiles_alive() == 3, "recharging hostile must count as alive"
        assert gm._wave_hostiles_total_health() > 0.0

    def test_force_eliminate_clears_low_battery_hostile(self):
        gm, bus, engine = _build_game_mode()
        h = engine.spawn_hostile()
        gm._wave_hostile_ids.add(h.target_id)
        h.status = "low_battery"
        gm._force_eliminate_wave_hostiles()
        assert h.status == "eliminated", "a recharging hostile must be force-eliminable"
