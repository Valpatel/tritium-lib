# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the strategic AI planner — faction-level decision-making."""

from __future__ import annotations

import pytest

from tritium_lib.sim_engine.ai.strategy import (
    STRATEGY_PROFILES,
    StrategicAI,
    StrategicGoal,
    StrategicPlan,
    StrategyProfile,
    _centroid,
    _cluster_radius,
    _enemy_concentrated,
    _flank_exposed,
    _gen_plan_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_world(
    friendly: list[dict] | None = None,
    enemies: list[dict] | None = None,
    objectives: list[dict] | None = None,
    terrain: dict | None = None,
    fog: bool = True,
    initial_strength: float | None = None,
) -> dict:
    """Build a minimal world_state dict for testing."""
    ws: dict = {
        "friendly_squads": friendly or [],
        "enemy_squads": enemies or [],
        "objectives": objectives or [],
        "terrain": terrain or {},
        "fog_of_war": fog,
    }
    if initial_strength is not None:
        ws["initial_friendly_strength"] = initial_strength
    return ws


def _squad(sid: str, pos: tuple, strength: float = 1.0, morale: float = 0.8, ammo: float = 0.8, specialty: str = "infantry") -> dict:
    return {"id": sid, "position": pos, "strength": strength, "morale": morale, "ammo": ammo, "specialty": specialty}


# ===========================================================================
# StrategicGoal enum
# ===========================================================================

class TestStrategicGoal:
    def test_all_goals_exist(self):
        expected = {"ATTACK", "DEFEND", "FLANK", "ENCIRCLE", "RETREAT",
                    "REINFORCE", "PROBE", "AMBUSH", "SIEGE", "PATROL"}
        assert set(g.name for g in StrategicGoal) == expected

    def test_goal_values(self):
        assert StrategicGoal.ATTACK.value == "attack"
        assert StrategicGoal.RETREAT.value == "retreat"
        assert StrategicGoal.ENCIRCLE.value == "encircle"

    def test_goal_count(self):
        assert len(StrategicGoal) == 10


# ===========================================================================
# StrategicPlan dataclass
# ===========================================================================

class TestStrategicPlan:
    def test_default_fields(self):
        plan = StrategicPlan(plan_id="p1", goal=StrategicGoal.ATTACK, faction="alpha")
        assert plan.primary_target is None
        assert plan.secondary_targets == []
        assert plan.assigned_squads == []
        assert plan.reserve_squads == []
        assert plan.priority == 0
        assert plan.confidence == 0.5
        assert plan.reasoning == ""

    def test_custom_fields(self):
        plan = StrategicPlan(
            plan_id="p2",
            goal=StrategicGoal.FLANK,
            faction="bravo",
            primary_target=(10.0, 20.0),
            secondary_targets=[(30.0, 40.0)],
            assigned_squads=["s1", "s2"],
            reserve_squads=["s3"],
            priority=8,
            confidence=0.9,
            reasoning="test reason",
        )
        assert plan.primary_target == (10.0, 20.0)
        assert len(plan.secondary_targets) == 1
        assert plan.confidence == 0.9

    def test_plan_independence(self):
        """Two plans don't share mutable defaults."""
        p1 = StrategicPlan(plan_id="a", goal=StrategicGoal.ATTACK, faction="f")
        p2 = StrategicPlan(plan_id="b", goal=StrategicGoal.DEFEND, faction="f")
        p1.assigned_squads.append("s1")
        assert "s1" not in p2.assigned_squads


# ===========================================================================
# StrategyProfile
# ===========================================================================

class TestStrategyProfile:
    def test_defaults(self):
        sp = StrategyProfile(name="test")
        assert sp.attack_threshold == 1.2
        assert sp.reserve_fraction == 0.2
        assert sp.probe_when_unknown is True

    def test_custom(self):
        sp = StrategyProfile(name="custom", aggression=0.9, caution=0.1)
        assert sp.aggression == 0.9
        assert sp.caution == 0.1


# ===========================================================================
# STRATEGY_PROFILES presets
# ===========================================================================

class TestStrategyProfiles:
    def test_all_profiles_present(self):
        expected = {"aggressive", "defensive", "balanced", "guerrilla", "blitz"}
        assert set(STRATEGY_PROFILES.keys()) == expected

    def test_aggressive_profile(self):
        p = STRATEGY_PROFILES["aggressive"]
        assert p.aggression > 0.7
        assert p.caution < 0.3
        assert p.attack_threshold < 1.0

    def test_defensive_profile(self):
        p = STRATEGY_PROFILES["defensive"]
        assert p.aggression < 0.4
        assert p.caution > 0.7
        assert p.attack_threshold > 1.5

    def test_guerrilla_profile(self):
        p = STRATEGY_PROFILES["guerrilla"]
        assert p.ambush_preference > 0.7
        assert p.flank_preference > 0.6

    def test_blitz_profile(self):
        p = STRATEGY_PROFILES["blitz"]
        assert p.aggression == 1.0
        assert p.reserve_fraction < 0.1
        assert p.siege_patience == 0.0

    def test_balanced_profile(self):
        p = STRATEGY_PROFILES["balanced"]
        assert 0.3 <= p.aggression <= 0.7
        assert 0.3 <= p.caution <= 0.7


# ===========================================================================
# Utility functions
# ===========================================================================

class TestHelpers:
    def test_centroid_empty(self):
        assert _centroid([]) == (0.0, 0.0)

    def test_centroid_single(self):
        assert _centroid([(5.0, 10.0)]) == (5.0, 10.0)

    def test_centroid_multiple(self):
        c = _centroid([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)])
        assert abs(c[0] - 5.0) < 0.01
        assert abs(c[1] - 5.0) < 0.01

    def test_gen_plan_id_unique(self):
        ids = {_gen_plan_id() for _ in range(100)}
        assert len(ids) == 100

    def test_gen_plan_id_format(self):
        pid = _gen_plan_id()
        assert pid.startswith("plan_")
        assert len(pid) == 13  # "plan_" + 8 hex chars

    def test_cluster_radius_single(self):
        assert _cluster_radius([(5.0, 5.0)]) == 0.0

    def test_cluster_radius_spread(self):
        pts = [(0.0, 0.0), (20.0, 0.0)]
        r = _cluster_radius(pts)
        assert r == pytest.approx(10.0, abs=0.1)

    def test_enemy_concentrated_few(self):
        assert _enemy_concentrated([(0.0, 0.0), (1.0, 1.0)]) is False

    def test_enemy_concentrated_tight(self):
        pts = [(0.0, 0.0), (3.0, 0.0), (0.0, 3.0), (3.0, 3.0)]
        assert _enemy_concentrated(pts, threshold=20.0) is True

    def test_enemy_concentrated_spread(self):
        pts = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)]
        assert _enemy_concentrated(pts, threshold=15.0) is False

    def test_flank_exposed_none_when_few_enemies(self):
        assert _flank_exposed([(0.0, 0.0)], [(50.0, 0.0)]) is None

    def test_flank_exposed_detects_spread_enemy(self):
        # Enemies spread along y-axis, friendlies approaching from x
        enemies = [(50.0, -20.0), (50.0, 20.0)]
        friendlies = [(0.0, 0.0)]
        result = _flank_exposed(enemies, friendlies)
        assert result is not None
        # Flank position should be offset from enemy center
        assert result != (50.0, 0.0)


# ===========================================================================
# StrategicAI construction
# ===========================================================================

class TestStrategicAIInit:
    def test_default_profile(self):
        ai = StrategicAI()
        assert ai.profile.name == "balanced"

    def test_string_profile(self):
        ai = StrategicAI("aggressive")
        assert ai.profile.name == "aggressive"

    def test_unknown_profile_falls_back(self):
        ai = StrategicAI("nonexistent")
        assert ai.profile.name == "balanced"

    def test_custom_profile_object(self):
        sp = StrategyProfile(name="custom", aggression=1.0)
        ai = StrategicAI(sp)
        assert ai.profile.name == "custom"
        assert ai.profile.aggression == 1.0

    def test_empty_plan_history(self):
        ai = StrategicAI()
        assert ai.plan_history == []


# ===========================================================================
# StrategicAI.assess
# ===========================================================================

class TestAssess:
    def test_empty_world(self):
        ai = StrategicAI()
        a = ai.assess(_make_world())
        assert a["force_ratio"] == pytest.approx(0.0, abs=0.01)
        assert a["num_friendly_squads"] == 0
        assert a["num_enemy_squads"] == 0

    def test_force_ratio(self):
        ai = StrategicAI()
        ws = _make_world(
            friendly=[_squad("s1", (0, 0), strength=10)],
            enemies=[_squad("e1", (50, 50), strength=5)],
        )
        a = ai.assess(ws)
        assert a["force_ratio"] == pytest.approx(2.0)

    def test_morale_average(self):
        ai = StrategicAI()
        ws = _make_world(friendly=[
            _squad("s1", (0, 0), morale=0.6),
            _squad("s2", (5, 5), morale=1.0),
        ])
        a = ai.assess(ws)
        assert a["friendly_morale"] == pytest.approx(0.8)

    def test_ammo_average(self):
        ai = StrategicAI()
        ws = _make_world(friendly=[
            _squad("s1", (0, 0), ammo=0.4),
            _squad("s2", (5, 5), ammo=0.8),
        ])
        a = ai.assess(ws)
        assert a["friendly_ammo"] == pytest.approx(0.6)

    def test_casualty_ratio(self):
        ai = StrategicAI()
        ws = _make_world(
            friendly=[_squad("s1", (0, 0), strength=5)],
            initial_strength=10.0,
        )
        a = ai.assess(ws)
        assert a["casualty_ratio"] == pytest.approx(0.5)

    def test_positions_extracted(self):
        ai = StrategicAI()
        ws = _make_world(
            friendly=[_squad("s1", (10, 20))],
            enemies=[_squad("e1", (50, 60))],
        )
        a = ai.assess(ws)
        assert (10, 20) in a["friendly_positions"]
        assert (50, 60) in a["enemy_positions"]

    def test_objectives_counted(self):
        ai = StrategicAI()
        ws = _make_world(objectives=[
            {"position": (10, 10), "owner": "friendly", "value": 5},
            {"position": (50, 50), "owner": "enemy", "value": 3},
            {"position": (30, 30), "owner": "contested", "value": 2},
        ])
        a = ai.assess(ws)
        assert a["friendly_objectives"] == 1
        assert a["enemy_objectives"] == 1
        assert a["contested_objectives"] == 1

    def test_terrain_passed_through(self):
        ai = StrategicAI()
        ws = _make_world(terrain={"chokepoints": [(25, 25)], "high_ground": [(30, 30)]})
        a = ai.assess(ws)
        assert len(a["chokepoints"]) == 1
        assert len(a["high_ground"]) == 1


# ===========================================================================
# StrategicAI.plan — decision tree
# ===========================================================================

class TestPlan:
    def test_retreat_on_heavy_casualties_and_low_ammo(self):
        ai = StrategicAI()
        a = {"force_ratio": 1.0, "friendly_morale": 0.5, "friendly_ammo": 0.2,
             "casualty_ratio": 0.6, "enemy_center": (50, 50), "exposed_flank": None,
             "enemy_concentrated": False, "fog_of_war": False, "num_enemy_squads": 3,
             "chokepoints": [], "enemy_objectives": 0, "front_distance": 50.0}
        plan = ai.plan("alpha", a)
        assert plan.goal == StrategicGoal.RETREAT
        assert plan.priority == 10

    def test_retreat_when_badly_outnumbered(self):
        ai = StrategicAI()  # balanced: retreat_threshold=0.4
        a = {"force_ratio": 0.3, "friendly_morale": 0.5, "friendly_ammo": 0.8,
             "casualty_ratio": 0.1, "enemy_center": (50, 50), "exposed_flank": None,
             "enemy_concentrated": False, "fog_of_war": False, "num_enemy_squads": 5,
             "chokepoints": [], "enemy_objectives": 0, "front_distance": 50.0}
        plan = ai.plan("alpha", a)
        assert plan.goal == StrategicGoal.RETREAT

    def test_defend_when_outnumbered(self):
        ai = StrategicAI()  # balanced: defend_threshold=0.8
        a = {"force_ratio": 0.6, "friendly_morale": 0.5, "friendly_ammo": 0.8,
             "casualty_ratio": 0.1, "enemy_center": (50, 50), "exposed_flank": None,
             "enemy_concentrated": False, "fog_of_war": False, "num_enemy_squads": 3,
             "chokepoints": [], "enemy_objectives": 0, "front_distance": 50.0}
        plan = ai.plan("alpha", a)
        assert plan.goal == StrategicGoal.DEFEND

    def test_probe_in_fog_no_enemies(self):
        ai = StrategicAI()
        a = {"force_ratio": 1.0, "friendly_morale": 0.8, "friendly_ammo": 0.8,
             "casualty_ratio": 0.0, "enemy_center": (0, 0), "exposed_flank": None,
             "enemy_concentrated": False, "fog_of_war": True, "num_enemy_squads": 0,
             "chokepoints": [], "enemy_objectives": 0, "front_distance": float("inf")}
        plan = ai.plan("alpha", a)
        assert plan.goal == StrategicGoal.PROBE

    def test_patrol_no_fog_no_enemies(self):
        ai = StrategicAI()
        a = {"force_ratio": 1.0, "friendly_morale": 0.8, "friendly_ammo": 0.8,
             "casualty_ratio": 0.0, "enemy_center": (0, 0), "exposed_flank": None,
             "enemy_concentrated": False, "fog_of_war": False, "num_enemy_squads": 0,
             "chokepoints": [], "enemy_objectives": 0, "front_distance": float("inf")}
        plan = ai.plan("alpha", a)
        assert plan.goal == StrategicGoal.PATROL

    def test_flank_when_exposed(self):
        ai = StrategicAI()
        a = {"force_ratio": 1.2, "friendly_morale": 0.8, "friendly_ammo": 0.8,
             "casualty_ratio": 0.0, "enemy_center": (50, 50),
             "exposed_flank": (70.0, 30.0),
             "enemy_concentrated": False, "fog_of_war": False, "num_enemy_squads": 3,
             "chokepoints": [], "enemy_objectives": 0, "front_distance": 50.0}
        plan = ai.plan("alpha", a)
        assert plan.goal == StrategicGoal.FLANK
        assert plan.primary_target == (70.0, 30.0)

    def test_encircle_concentrated_enemy(self):
        ai = StrategicAI()
        a = {"force_ratio": 1.5, "friendly_morale": 0.8, "friendly_ammo": 0.8,
             "casualty_ratio": 0.0, "enemy_center": (50, 50), "exposed_flank": None,
             "enemy_concentrated": True, "fog_of_war": False, "num_enemy_squads": 3,
             "chokepoints": [], "enemy_objectives": 0, "front_distance": 50.0}
        plan = ai.plan("alpha", a)
        assert plan.goal == StrategicGoal.ENCIRCLE

    def test_ambush_guerrilla_with_chokepoints(self):
        ai = StrategicAI("guerrilla")
        a = {"force_ratio": 1.0, "friendly_morale": 0.8, "friendly_ammo": 0.8,
             "casualty_ratio": 0.0, "enemy_center": (50, 50), "exposed_flank": None,
             "enemy_concentrated": False, "fog_of_war": False, "num_enemy_squads": 3,
             "chokepoints": [(40.0, 45.0)], "enemy_objectives": 0, "front_distance": 50.0}
        plan = ai.plan("alpha", a)
        assert plan.goal == StrategicGoal.AMBUSH
        assert plan.primary_target == (40.0, 45.0)

    def test_siege_enemy_holds_objective(self):
        ai = StrategicAI("balanced")
        a = {"force_ratio": 1.3, "friendly_morale": 0.8, "friendly_ammo": 0.8,
             "casualty_ratio": 0.0, "enemy_center": (50, 50), "exposed_flank": None,
             "enemy_concentrated": False, "fog_of_war": False, "num_enemy_squads": 3,
             "chokepoints": [], "enemy_objectives": 2, "front_distance": 50.0}
        plan = ai.plan("alpha", a)
        # balanced has siege_patience=0.5 > 0.4, ratio=1.3 >= 1.0
        assert plan.goal == StrategicGoal.SIEGE

    def test_attack_when_superior(self):
        ai = StrategicAI()
        a = {"force_ratio": 1.5, "friendly_morale": 0.9, "friendly_ammo": 0.9,
             "casualty_ratio": 0.0, "enemy_center": (50, 50), "exposed_flank": None,
             "enemy_concentrated": False, "fog_of_war": False, "num_enemy_squads": 3,
             "chokepoints": [], "enemy_objectives": 0, "front_distance": 50.0}
        plan = ai.plan("alpha", a)
        assert plan.goal == StrategicGoal.ATTACK

    def test_default_defend(self):
        ai = StrategicAI()
        # Force ratio between defend_threshold and attack_threshold, no special conditions
        a = {"force_ratio": 1.0, "friendly_morale": 0.8, "friendly_ammo": 0.8,
             "casualty_ratio": 0.0, "enemy_center": (50, 50), "exposed_flank": None,
             "enemy_concentrated": False, "fog_of_war": False, "num_enemy_squads": 3,
             "chokepoints": [], "enemy_objectives": 0, "front_distance": 50.0}
        plan = ai.plan("alpha", a)
        assert plan.goal == StrategicGoal.DEFEND

    def test_plan_has_id(self):
        ai = StrategicAI()
        a = {"force_ratio": 2.0, "friendly_morale": 0.8, "friendly_ammo": 0.8,
             "casualty_ratio": 0.0, "enemy_center": (50, 50), "exposed_flank": None,
             "enemy_concentrated": False, "fog_of_war": False, "num_enemy_squads": 1,
             "chokepoints": [], "enemy_objectives": 0, "front_distance": 50.0}
        plan = ai.plan("alpha", a)
        assert plan.plan_id.startswith("plan_")

    def test_plan_stored_in_history(self):
        ai = StrategicAI()
        a = {"force_ratio": 2.0, "friendly_morale": 0.8, "friendly_ammo": 0.8,
             "casualty_ratio": 0.0, "enemy_center": (50, 50), "exposed_flank": None,
             "enemy_concentrated": False, "fog_of_war": False, "num_enemy_squads": 1,
             "chokepoints": [], "enemy_objectives": 0, "front_distance": 50.0}
        plan = ai.plan("alpha", a)
        assert len(ai.plan_history) == 1
        assert ai.plan_history[0].plan_id == plan.plan_id

    def test_plan_reasoning_nonempty(self):
        ai = StrategicAI()
        a = {"force_ratio": 0.3, "friendly_morale": 0.5, "friendly_ammo": 0.8,
             "casualty_ratio": 0.1, "enemy_center": (50, 50), "exposed_flank": None,
             "enemy_concentrated": False, "fog_of_war": False, "num_enemy_squads": 5,
             "chokepoints": [], "enemy_objectives": 0, "front_distance": 50.0}
        plan = ai.plan("alpha", a)
        assert len(plan.reasoning) > 0

    def test_blitz_attacks_easily(self):
        ai = StrategicAI("blitz")  # attack_threshold=0.7
        a = {"force_ratio": 0.8, "friendly_morale": 0.8, "friendly_ammo": 0.8,
             "casualty_ratio": 0.0, "enemy_center": (50, 50), "exposed_flank": None,
             "enemy_concentrated": False, "fog_of_war": False, "num_enemy_squads": 3,
             "chokepoints": [], "enemy_objectives": 0, "front_distance": 50.0}
        plan = ai.plan("alpha", a)
        assert plan.goal == StrategicGoal.ATTACK

    def test_defensive_needs_big_advantage_to_attack(self):
        ai = StrategicAI("defensive")  # attack_threshold=2.0
        a = {"force_ratio": 1.8, "friendly_morale": 0.8, "friendly_ammo": 0.8,
             "casualty_ratio": 0.0, "enemy_center": (50, 50), "exposed_flank": None,
             "enemy_concentrated": False, "fog_of_war": False, "num_enemy_squads": 3,
             "chokepoints": [], "enemy_objectives": 0, "front_distance": 50.0}
        plan = ai.plan("alpha", a)
        # 1.8 < 2.0, so should NOT attack; should siege or defend
        assert plan.goal != StrategicGoal.ATTACK


# ===========================================================================
# StrategicAI.adapt
# ===========================================================================

class TestAdapt:
    def _base_plan(self, goal: StrategicGoal = StrategicGoal.ATTACK) -> StrategicPlan:
        return StrategicPlan(
            plan_id="orig",
            goal=goal,
            faction="alpha",
            primary_target=(50.0, 50.0),
            secondary_targets=[(70.0, 70.0)],
            assigned_squads=["s1", "s2"],
            confidence=0.7,
            priority=8,
            reasoning="Initial plan",
        )

    def test_adapt_no_events(self):
        ai = StrategicAI()
        plan = self._base_plan()
        adapted = ai.adapt(plan, [])
        assert adapted.goal == StrategicGoal.ATTACK
        assert adapted.plan_id != plan.plan_id

    def test_ambush_detected_during_attack(self):
        ai = StrategicAI()
        plan = self._base_plan(StrategicGoal.ATTACK)
        adapted = ai.adapt(plan, [{"type": "ambush_detected"}])
        assert adapted.goal == StrategicGoal.DEFEND
        assert adapted.confidence < plan.confidence

    def test_ambush_detected_during_flank(self):
        ai = StrategicAI()
        plan = self._base_plan(StrategicGoal.FLANK)
        adapted = ai.adapt(plan, [{"type": "ambush_detected"}])
        assert adapted.goal == StrategicGoal.RETREAT

    def test_reinforcements_escalate_defend_to_attack(self):
        ai = StrategicAI()  # balanced: aggression=0.5 > 0.4
        plan = self._base_plan(StrategicGoal.DEFEND)
        adapted = ai.adapt(plan, [{"type": "reinforcements_arrived", "strength": 5.0}])
        assert adapted.goal == StrategicGoal.ATTACK
        assert adapted.confidence > plan.confidence

    def test_reinforcements_cautious_stays_defensive(self):
        ai = StrategicAI("defensive")  # aggression=0.2 < 0.4
        plan = self._base_plan(StrategicGoal.DEFEND)
        adapted = ai.adapt(plan, [{"type": "reinforcements_arrived", "strength": 5.0}])
        assert adapted.goal == StrategicGoal.DEFEND

    def test_objective_taken_shifts_target(self):
        ai = StrategicAI()
        plan = self._base_plan()
        adapted = ai.adapt(plan, [{"type": "objective_taken", "position": (50.0, 50.0)}])
        # Primary was (50,50), secondary was [(70,70)], so new primary should be (70,70)
        assert adapted.primary_target == (70.0, 70.0)
        assert len(adapted.secondary_targets) == 0

    def test_objective_taken_all_done(self):
        ai = StrategicAI()
        plan = StrategicPlan(
            plan_id="orig", goal=StrategicGoal.ATTACK, faction="alpha",
            primary_target=(50.0, 50.0), secondary_targets=[], confidence=0.7,
        )
        adapted = ai.adapt(plan, [{"type": "objective_taken", "position": (50.0, 50.0)}])
        assert adapted.goal == StrategicGoal.DEFEND

    def test_heavy_casualties_forces_retreat(self):
        ai = StrategicAI()
        plan = self._base_plan()
        adapted = ai.adapt(plan, [{"type": "heavy_casualties", "casualty_percent": 0.5}])
        assert adapted.goal == StrategicGoal.RETREAT
        assert adapted.priority == 10

    def test_moderate_casualties_reduce_confidence(self):
        ai = StrategicAI()
        plan = self._base_plan()
        adapted = ai.adapt(plan, [{"type": "heavy_casualties", "casualty_percent": 0.2}])
        assert adapted.confidence < plan.confidence
        # Not forced retreat with only 20% casualties
        assert adapted.goal == StrategicGoal.ATTACK

    def test_enemy_retreating_pursue(self):
        ai = StrategicAI()  # balanced: aggression=0.5 > 0.3
        plan = self._base_plan(StrategicGoal.DEFEND)
        adapted = ai.adapt(plan, [{"type": "enemy_retreating", "position": (80.0, 80.0)}])
        assert adapted.goal == StrategicGoal.ATTACK
        assert adapted.primary_target == (80.0, 80.0)

    def test_enemy_retreating_cautious_patrol(self):
        ai = StrategicAI(StrategyProfile(name="timid", aggression=0.1))
        plan = self._base_plan(StrategicGoal.DEFEND)
        adapted = ai.adapt(plan, [{"type": "enemy_retreating"}])
        assert adapted.goal == StrategicGoal.PATROL

    def test_flank_threatened(self):
        ai = StrategicAI()
        plan = self._base_plan(StrategicGoal.ATTACK)
        adapted = ai.adapt(plan, [{"type": "flank_threatened", "position": (30.0, 80.0)}])
        assert adapted.goal == StrategicGoal.DEFEND
        assert (30.0, 80.0) in adapted.secondary_targets

    def test_supply_line_cut(self):
        ai = StrategicAI()
        plan = self._base_plan()
        adapted = ai.adapt(plan, [{"type": "supply_line_cut"}])
        assert adapted.goal == StrategicGoal.RETREAT
        assert adapted.priority == 9

    def test_intel_update_adds_positions(self):
        ai = StrategicAI()
        plan = self._base_plan()
        adapted = ai.adapt(plan, [{"type": "intel_update", "enemy_positions": [(90.0, 10.0), (95.0, 15.0)]}])
        assert (90.0, 10.0) in adapted.secondary_targets
        assert (95.0, 15.0) in adapted.secondary_targets

    def test_multiple_events_processed(self):
        ai = StrategicAI()
        plan = self._base_plan()
        events = [
            {"type": "intel_update", "enemy_positions": [(90.0, 10.0)]},
            {"type": "reinforcements_arrived", "strength": 3.0},
        ]
        adapted = ai.adapt(plan, events)
        assert (90.0, 10.0) in adapted.secondary_targets
        assert adapted.confidence > plan.confidence  # reinforcements boost

    def test_adapt_stores_in_history(self):
        ai = StrategicAI()
        plan = self._base_plan()
        adapted = ai.adapt(plan, [{"type": "supply_line_cut"}])
        assert len(ai.plan_history) == 1
        assert ai.plan_history[0].plan_id == adapted.plan_id

    def test_adapt_does_not_mutate_original(self):
        ai = StrategicAI()
        plan = self._base_plan()
        original_goal = plan.goal
        original_target = plan.primary_target
        ai.adapt(plan, [{"type": "supply_line_cut"}])
        assert plan.goal == original_goal
        assert plan.primary_target == original_target

    def test_confidence_clamped(self):
        ai = StrategicAI()
        plan = self._base_plan()
        # Stack many positive events
        events = [{"type": "reinforcements_arrived", "strength": 10.0}] * 10
        adapted = ai.adapt(plan, events)
        assert adapted.confidence <= 1.0


# ===========================================================================
# StrategicAI.assign_squads
# ===========================================================================

class TestAssignSquads:
    def _squads(self, n: int) -> list[dict]:
        return [_squad(f"sq{i}", (i * 10.0, 0.0), strength=10 - i) for i in range(n)]

    def _positions(self, n: int) -> dict:
        return {f"sq{i}": (i * 10.0, 0.0) for i in range(n)}

    def test_empty_squads(self):
        ai = StrategicAI()
        plan = StrategicPlan(plan_id="p", goal=StrategicGoal.ATTACK, faction="a")
        result = ai.assign_squads(plan, [], {})
        assert result == {}

    def test_attack_assignment(self):
        ai = StrategicAI()
        plan = StrategicPlan(plan_id="p", goal=StrategicGoal.ATTACK, faction="a",
                             primary_target=(100.0, 0.0))
        squads = self._squads(4)
        pos = self._positions(4)
        result = ai.assign_squads(plan, squads, pos)
        roles = {v["role"] for v in result.values()}
        assert "assault" in roles
        assert "support" in roles or "reserve" in roles

    def test_defend_assignment(self):
        ai = StrategicAI()
        plan = StrategicPlan(plan_id="p", goal=StrategicGoal.DEFEND, faction="a")
        squads = self._squads(3)
        pos = self._positions(3)
        result = ai.assign_squads(plan, squads, pos)
        for v in result.values():
            assert v["role"] in ("defense", "reserve")

    def test_flank_assignment(self):
        ai = StrategicAI()
        plan = StrategicPlan(plan_id="p", goal=StrategicGoal.FLANK, faction="a",
                             primary_target=(80.0, 20.0),
                             secondary_targets=[(50.0, 50.0)])
        squads = self._squads(6)
        pos = self._positions(6)
        result = ai.assign_squads(plan, squads, pos)
        roles = [v["role"] for v in result.values()]
        assert "flanking" in roles
        assert "support" in roles or "reserve" in roles

    def test_encircle_positions_surround_target(self):
        ai = StrategicAI()
        plan = StrategicPlan(plan_id="p", goal=StrategicGoal.ENCIRCLE, faction="a",
                             primary_target=(50.0, 50.0))
        squads = self._squads(4)
        pos = self._positions(4)
        result = ai.assign_squads(plan, squads, pos)
        targets = [v["target"] for v in result.values() if v["role"] == "encircle"]
        # All encircle targets should be roughly 40m from (50,50)
        from tritium_lib.sim_engine.ai.steering import distance as dist
        for t in targets:
            d = dist(t, (50.0, 50.0))
            assert 35.0 < d < 45.0

    def test_retreat_has_rearguard(self):
        ai = StrategicAI()
        plan = StrategicPlan(plan_id="p", goal=StrategicGoal.RETREAT, faction="a")
        squads = self._squads(3)
        pos = self._positions(3)
        result = ai.assign_squads(plan, squads, pos)
        roles = [v["role"] for v in result.values()]
        assert "rearguard" in roles

    def test_probe_assignment(self):
        ai = StrategicAI()
        plan = StrategicPlan(plan_id="p", goal=StrategicGoal.PROBE, faction="a",
                             primary_target=(100.0, 100.0))
        squads = self._squads(4)
        pos = self._positions(4)
        result = ai.assign_squads(plan, squads, pos)
        roles = [v["role"] for v in result.values()]
        assert "probe" in roles

    def test_ambush_assignment(self):
        ai = StrategicAI()
        plan = StrategicPlan(plan_id="p", goal=StrategicGoal.AMBUSH, faction="a",
                             primary_target=(50.0, 50.0))
        squads = self._squads(3)
        pos = self._positions(3)
        result = ai.assign_squads(plan, squads, pos)
        for v in result.values():
            assert v["role"] in ("ambush", "reserve")

    def test_siege_assignment(self):
        ai = StrategicAI()
        plan = StrategicPlan(plan_id="p", goal=StrategicGoal.SIEGE, faction="a",
                             primary_target=(50.0, 50.0))
        squads = self._squads(4)
        pos = self._positions(4)
        result = ai.assign_squads(plan, squads, pos)
        roles = [v["role"] for v in result.values()]
        assert "siege" in roles

    def test_patrol_assignment(self):
        ai = StrategicAI()
        plan = StrategicPlan(plan_id="p", goal=StrategicGoal.PATROL, faction="a")
        squads = self._squads(3)
        pos = self._positions(3)
        result = ai.assign_squads(plan, squads, pos)
        for v in result.values():
            assert v["role"] in ("patrol", "reserve")

    def test_reserve_fraction(self):
        """With 10 squads and 20% reserve, 2 should be in reserve."""
        ai = StrategicAI()  # balanced: reserve_fraction=0.2
        plan = StrategicPlan(plan_id="p", goal=StrategicGoal.ATTACK, faction="a",
                             primary_target=(100.0, 0.0))
        squads = self._squads(10)
        pos = self._positions(10)
        result = ai.assign_squads(plan, squads, pos)
        reserves = [sid for sid, v in result.items() if v["role"] == "reserve"]
        assert len(reserves) == 2

    def test_blitz_minimal_reserves(self):
        ai = StrategicAI("blitz")  # reserve_fraction=0.05
        plan = StrategicPlan(plan_id="p", goal=StrategicGoal.ATTACK, faction="a",
                             primary_target=(100.0, 0.0))
        squads = self._squads(10)
        pos = self._positions(10)
        result = ai.assign_squads(plan, squads, pos)
        reserves = [sid for sid, v in result.items() if v["role"] == "reserve"]
        assert len(reserves) <= 1

    def test_all_squads_assigned(self):
        ai = StrategicAI()
        plan = StrategicPlan(plan_id="p", goal=StrategicGoal.DEFEND, faction="a")
        squads = self._squads(5)
        pos = self._positions(5)
        result = ai.assign_squads(plan, squads, pos)
        assert len(result) == 5

    def test_plan_lists_updated(self):
        ai = StrategicAI()
        plan = StrategicPlan(plan_id="p", goal=StrategicGoal.ATTACK, faction="a",
                             primary_target=(100.0, 0.0))
        squads = self._squads(5)
        pos = self._positions(5)
        ai.assign_squads(plan, squads, pos)
        assert len(plan.assigned_squads) + len(plan.reserve_squads) == 5

    def test_reinforce_assignment(self):
        ai = StrategicAI()
        plan = StrategicPlan(plan_id="p", goal=StrategicGoal.REINFORCE, faction="a",
                             primary_target=(80.0, 80.0))
        squads = self._squads(3)
        pos = self._positions(3)
        result = ai.assign_squads(plan, squads, pos)
        for v in result.values():
            assert v["role"] in ("reinforce", "reserve")

    def test_orders_are_strings(self):
        ai = StrategicAI()
        plan = StrategicPlan(plan_id="p", goal=StrategicGoal.ATTACK, faction="a",
                             primary_target=(100.0, 0.0))
        squads = self._squads(3)
        pos = self._positions(3)
        result = ai.assign_squads(plan, squads, pos)
        for v in result.values():
            assert isinstance(v["orders"], str)
            assert len(v["orders"]) > 0


# ===========================================================================
# Integration: assess -> plan -> assign
# ===========================================================================

class TestIntegration:
    def test_full_pipeline_attack(self):
        ai = StrategicAI("aggressive")
        ws = _make_world(
            friendly=[_squad(f"s{i}", (i * 10.0, 0.0), strength=5) for i in range(5)],
            enemies=[_squad(f"e{i}", (100 + i * 5.0, 50.0), strength=2) for i in range(3)],
            fog=False,
        )
        assessment = ai.assess(ws)
        plan = ai.plan("alpha", assessment)
        positions = {f"s{i}": (i * 10.0, 0.0) for i in range(5)}
        squads = [_squad(f"s{i}", (i * 10.0, 0.0), strength=5) for i in range(5)]
        assignments = ai.assign_squads(plan, squads, positions)
        assert len(assignments) == 5
        assert plan.faction == "alpha"

    def test_full_pipeline_retreat(self):
        ai = StrategicAI()
        ws = _make_world(
            friendly=[_squad("s0", (0, 0), strength=2, ammo=0.2)],
            enemies=[_squad(f"e{i}", (30 + i * 5.0, 30.0), strength=5) for i in range(5)],
            fog=False,
            initial_strength=10.0,
        )
        assessment = ai.assess(ws)
        plan = ai.plan("alpha", assessment)
        assert plan.goal == StrategicGoal.RETREAT

    def test_full_pipeline_adapt(self):
        ai = StrategicAI()
        ws = _make_world(
            friendly=[_squad(f"s{i}", (i * 10.0, 0.0), strength=5) for i in range(4)],
            enemies=[_squad(f"e{i}", (80.0, i * 10.0), strength=3) for i in range(3)],
            fog=False,
        )
        assessment = ai.assess(ws)
        plan = ai.plan("alpha", assessment)
        adapted = ai.adapt(plan, [
            {"type": "reinforcements_arrived", "strength": 10.0},
            {"type": "intel_update", "enemy_positions": [(120.0, 50.0)]},
        ])
        assert adapted.plan_id != plan.plan_id
        assert (120.0, 50.0) in adapted.secondary_targets
        assert len(ai.plan_history) == 2  # original plan + adapted

    def test_history_accumulates(self):
        ai = StrategicAI()
        a = {"force_ratio": 2.0, "friendly_morale": 0.8, "friendly_ammo": 0.8,
             "casualty_ratio": 0.0, "enemy_center": (50, 50), "exposed_flank": None,
             "enemy_concentrated": False, "fog_of_war": False, "num_enemy_squads": 1,
             "chokepoints": [], "enemy_objectives": 0, "front_distance": 50.0}
        for _ in range(5):
            ai.plan("alpha", a)
        assert len(ai.plan_history) == 5

    def test_different_profiles_different_decisions(self):
        """Same situation, different profiles should produce different plans."""
        a = {"force_ratio": 0.8, "friendly_morale": 0.8, "friendly_ammo": 0.8,
             "casualty_ratio": 0.0, "enemy_center": (50, 50), "exposed_flank": None,
             "enemy_concentrated": False, "fog_of_war": False, "num_enemy_squads": 3,
             "chokepoints": [], "enemy_objectives": 0, "front_distance": 50.0}
        blitz_plan = StrategicAI("blitz").plan("a", a)
        defensive_plan = StrategicAI("defensive").plan("a", a)
        assert blitz_plan.goal != defensive_plan.goal
