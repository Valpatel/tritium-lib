# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the combat AI tactics engine.

Tests cover threat assessment, situation evaluation, decision-making,
personality effects, and squad coordination.
"""

from __future__ import annotations

import pytest

from tritium_lib.sim_engine.ai.steering import Vec2, distance
from tritium_lib.sim_engine.ai.tactics import (
    AIPersonality,
    PERSONALITY_PRESETS,
    TacticalAction,
    TacticalSituation,
    TacticsEngine,
    ThreatAssessment,
    _centroid,
    _cluster_radius,
    _nearest_threat,
    _highest_threat,
    _threats_clustered,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> TacticsEngine:
    return TacticsEngine()


@pytest.fixture
def veteran_engine() -> TacticsEngine:
    return TacticsEngine(personality=PERSONALITY_PRESETS["veteran"])


@pytest.fixture
def recruit_engine() -> TacticsEngine:
    return TacticsEngine(personality=PERSONALITY_PRESETS["recruit"])


@pytest.fixture
def berserker_engine() -> TacticsEngine:
    return TacticsEngine(personality=PERSONALITY_PRESETS["berserker"])


@pytest.fixture
def sniper_engine() -> TacticsEngine:
    return TacticsEngine(personality=PERSONALITY_PRESETS["sniper"])


@pytest.fixture
def medic_engine() -> TacticsEngine:
    return TacticsEngine(personality=PERSONALITY_PRESETS["medic"])


@pytest.fixture
def leader_engine() -> TacticsEngine:
    return TacticsEngine(personality=PERSONALITY_PRESETS["leader"])


def _make_enemy(
    eid: str = "e1",
    pos: Vec2 = (50.0, 50.0),
    damage: float = 5.0,
    facing: float = 0.0,
    suppressing: bool = False,
    health: float = 1.0,
    last_seen: float = 0.0,
) -> dict:
    return {
        "id": eid,
        "pos": pos,
        "damage": damage,
        "facing": facing,
        "suppressing": suppressing,
        "health": health,
        "last_seen": last_seen,
    }


def _make_unit(
    uid: str = "u1",
    pos: Vec2 = (0.0, 0.0),
    health: float = 1.0,
    ammo: float = 1.0,
    morale: float = 1.0,
    facing: float = 0.0,
    squad_order: str | None = None,
    role: str = "rifleman",
) -> dict:
    return {
        "id": uid,
        "pos": pos,
        "health": health,
        "ammo": ammo,
        "morale": morale,
        "facing": facing,
        "squad_order": squad_order,
        "role": role,
    }


# ---------------------------------------------------------------------------
# ThreatAssessment tests
# ---------------------------------------------------------------------------


class TestThreatAssessment:
    """Tests for threat assessment ranking and evaluation."""

    def test_assess_single_enemy(self, engine: TacticsEngine):
        enemies = [_make_enemy("e1", (50, 50), damage=5.0)]
        threats = engine.assess_threats((0, 0), enemies, dt=0.1)
        assert len(threats) == 1
        assert threats[0].threat_id == "e1"
        assert threats[0].distance == pytest.approx(distance((0, 0), (50, 50)), abs=0.1)

    def test_assess_multiple_enemies(self, engine: TacticsEngine):
        enemies = [
            _make_enemy("e1", (10, 0), damage=5.0),
            _make_enemy("e2", (50, 0), damage=5.0),
            _make_enemy("e3", (30, 0), damage=5.0),
        ]
        threats = engine.assess_threats((0, 0), enemies, dt=0.1)
        assert len(threats) == 3

    def test_closer_enemy_higher_threat(self, engine: TacticsEngine):
        """Closer enemies should have higher threat level (same damage)."""
        enemies = [
            _make_enemy("close", (10, 0), damage=5.0),
            _make_enemy("far", (80, 0), damage=5.0),
        ]
        threats = engine.assess_threats((0, 0), enemies, dt=0.1)
        close_threat = next(t for t in threats if t.threat_id == "close")
        far_threat = next(t for t in threats if t.threat_id == "far")
        assert close_threat.threat_level > far_threat.threat_level

    def test_higher_damage_higher_threat(self, engine: TacticsEngine):
        """Higher damage enemies should have higher threat level (same distance)."""
        enemies = [
            _make_enemy("weak", (30, 0), damage=2.0),
            _make_enemy("strong", (30, 0), damage=8.0),
        ]
        threats = engine.assess_threats((0, 0), enemies, dt=0.1)
        weak_threat = next(t for t in threats if t.threat_id == "weak")
        strong_threat = next(t for t in threats if t.threat_id == "strong")
        assert strong_threat.threat_level > weak_threat.threat_level

    def test_threats_sorted_by_threat_level(self, engine: TacticsEngine):
        """Threats should be sorted descending by threat_level."""
        enemies = [
            _make_enemy("e1", (10, 0), damage=1.0),
            _make_enemy("e2", (5, 0), damage=10.0),
            _make_enemy("e3", (90, 0), damage=3.0),
        ]
        threats = engine.assess_threats((0, 0), enemies, dt=0.1)
        for i in range(len(threats) - 1):
            assert threats[i].threat_level >= threats[i + 1].threat_level

    def test_threat_level_range(self, engine: TacticsEngine):
        """Threat levels should be between 0 and 1."""
        enemies = [
            _make_enemy("e1", (1, 0), damage=100.0),
            _make_enemy("e2", (200, 0), damage=0.1),
        ]
        threats = engine.assess_threats((0, 0), enemies, dt=0.1)
        for t in threats:
            assert 0.0 <= t.threat_level <= 1.0

    def test_flanking_detection(self, engine: TacticsEngine):
        """Enemy whose frontal arc does NOT include the unit should be flagged as flanking.

        is_flanking(attacker_pos=unit, target_pos=enemy, target_facing) returns True
        when the unit is outside the enemy's frontal arc — i.e., the unit IS flanking
        the enemy.  We set the enemy to face *toward* the unit (facing=pi, pointing
        left toward origin) so the unit is IN the frontal arc -> not flanking.
        To get is_flanking=True, the enemy must face *away* from the unit (facing=0,
        pointing right, away from origin).
        """
        # Enemy at (50,0) facing right (0 rad = +x), unit at origin is behind them
        enemies = [_make_enemy("e1", (50, 0), facing=0.0)]
        threats = engine.assess_threats((0, 0), enemies, dt=0.1)
        assert threats[0].is_flanking

    def test_suppression_flag(self, engine: TacticsEngine):
        enemies = [_make_enemy("e1", (30, 0), suppressing=True)]
        threats = engine.assess_threats((0, 0), enemies, dt=0.1)
        assert threats[0].is_suppressing

    def test_health_estimate(self, engine: TacticsEngine):
        enemies = [_make_enemy("e1", (30, 0), health=0.3)]
        threats = engine.assess_threats((0, 0), enemies, dt=0.1)
        assert threats[0].estimated_health == pytest.approx(0.3)

    def test_empty_enemies(self, engine: TacticsEngine):
        threats = engine.assess_threats((0, 0), [], dt=0.1)
        assert threats == []


# ---------------------------------------------------------------------------
# Situation evaluation tests
# ---------------------------------------------------------------------------


class TestSituationEvaluation:
    """Tests for tactical situation building."""

    def test_basic_situation(self, engine: TacticsEngine):
        unit = _make_unit(pos=(0, 0), health=0.8, ammo=0.5, morale=0.7)
        threats = engine.assess_threats((0, 0), [_make_enemy()], dt=0.1)
        allies = [{"pos": (5, 0)}, {"pos": (10, 0)}]
        sit = engine.evaluate_situation(unit, threats, allies, cover=[(20, 0)])
        assert sit.unit_pos == (0, 0)
        assert sit.unit_health == pytest.approx(0.8)
        assert sit.unit_ammo == pytest.approx(0.5)
        assert sit.unit_morale == pytest.approx(0.7)
        assert sit.allies_nearby == 2
        assert len(sit.threats) == 1

    def test_allies_count_within_range(self, engine: TacticsEngine):
        unit = _make_unit(pos=(0, 0))
        allies = [{"pos": (10, 0)}, {"pos": (20, 0)}, {"pos": (100, 0)}]
        sit = engine.evaluate_situation(unit, [], allies, cover=[])
        assert sit.allies_nearby == 2  # third ally is too far (100m)

    def test_no_allies(self, engine: TacticsEngine):
        unit = _make_unit(pos=(0, 0))
        sit = engine.evaluate_situation(unit, [], [], cover=[])
        assert sit.allies_nearby == 0

    def test_squad_order_passed_through(self, engine: TacticsEngine):
        unit = _make_unit(pos=(0, 0), squad_order="advance")
        sit = engine.evaluate_situation(unit, [], [], cover=[])
        assert sit.squad_order == "advance"

    def test_cover_positions_stored(self, engine: TacticsEngine):
        unit = _make_unit(pos=(0, 0))
        cover = [(10, 5), (20, -5), (15, 0)]
        sit = engine.evaluate_situation(unit, [], [], cover=cover)
        assert sit.cover_positions == cover


# ---------------------------------------------------------------------------
# Decision-making tests
# ---------------------------------------------------------------------------


class TestDecisionMaking:
    """Tests for the core decide_action decision tree."""

    def test_low_health_triggers_retreat(self, engine: TacticsEngine):
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.1, unit_ammo=0.8, unit_morale=0.8,
            threats=[ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)],
            allies_nearby=2, in_cover=True, cover_positions=[(10, 10)],
            has_los_to_threats=[True], squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type == "retreat"
        assert "health" in action.reasoning.lower() or "Health" in action.reasoning

    def test_low_ammo_triggers_retreat(self, engine: TacticsEngine):
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.05, unit_morale=0.8,
            threats=[ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)],
            allies_nearby=2, in_cover=True, cover_positions=[(10, 10)],
            has_los_to_threats=[True], squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type == "retreat"
        assert "ammo" in action.reasoning.lower() or "Ammo" in action.reasoning

    def test_low_morale_triggers_retreat(self, engine: TacticsEngine):
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.1,
            threats=[ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)],
            allies_nearby=2, in_cover=True, cover_positions=[(10, 10)],
            has_los_to_threats=[True], squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type == "retreat"
        assert "morale" in action.reasoning.lower() or "Morale" in action.reasoning

    def test_suppressed_stays_in_cover(self, engine: TacticsEngine):
        # All threats suppressing -> suppression ratio = 1.0 > 0.7
        threats = [
            ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, True, 0.0, 1.0),
            ThreatAssessment("e2", (60, 0), 60.0, 0.4, False, True, 0.0, 1.0),
        ]
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=threats, allies_nearby=2, in_cover=True,
            cover_positions=[(10, 10)], has_los_to_threats=[True, True],
            squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type == "hold"
        assert "suppression" in action.reasoning.lower()

    def test_suppressed_not_in_cover_seeks_cover(self, engine: TacticsEngine):
        threats = [
            ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, True, 0.0, 1.0),
        ]
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=threats, allies_nearby=2, in_cover=False,
            cover_positions=[(10, 10)], has_los_to_threats=[True],
            squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type == "take_cover"

    def test_no_threats_advance_or_hold(self, engine: TacticsEngine):
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=[], allies_nearby=2, in_cover=False,
            cover_positions=[], has_los_to_threats=[],
            squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type in ("advance", "hold")
        assert "no threats" in action.reasoning.lower()

    def test_flanking_threat_triggers_relocate(self, engine: TacticsEngine):
        threats = [
            ThreatAssessment("e1", (50, 0), 50.0, 0.5, True, False, 0.0, 1.0),
        ]
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=threats, allies_nearby=2, in_cover=True,
            cover_positions=[(10, 10), (-10, 5)],
            has_los_to_threats=[True], squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type == "relocate"
        assert "flank" in action.reasoning.lower()

    def test_outnumbered_3to1_triggers_suppress_or_retreat(self, engine: TacticsEngine):
        # 6 enemies vs 1 unit + 1 ally = 6:2, ratio = 3:1
        threats = [
            ThreatAssessment(f"e{i}", (50 + i * 5, 0), 50.0 + i * 5, 0.4, False, False, 0.0, 1.0)
            for i in range(6)
        ]
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=threats, allies_nearby=1, in_cover=True,
            cover_positions=[(10, 10)], has_los_to_threats=[True] * 6,
            squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type in ("suppress", "retreat")

    def test_in_cover_with_los_engages(self, engine: TacticsEngine):
        threats = [
            ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0),
        ]
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=threats, allies_nearby=3, in_cover=True,
            cover_positions=[(10, 10)], has_los_to_threats=[True],
            squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type == "engage"
        assert action.target_id == "e1"

    def test_not_in_cover_takes_cover(self, engine: TacticsEngine):
        threats = [
            ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0),
        ]
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=threats, allies_nearby=3, in_cover=False,
            cover_positions=[(10, 10)], has_los_to_threats=[True],
            squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type == "take_cover"

    def test_clustered_enemies_grenade(self, engine: TacticsEngine):
        """Clustered enemies should trigger grenade decision when exposed and aggressive."""
        # Unit is exposed (no cover positions), not in cover, aggressive enough
        # Threats are NOT flanking, NOT suppressing, within 10m of each other
        threats = [
            ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0),
            ThreatAssessment("e2", (52, 2), 52.0, 0.4, False, False, 0.0, 1.0),
            ThreatAssessment("e3", (51, -1), 51.0, 0.45, False, False, 0.0, 1.0),
        ]
        # Use berserker personality (aggression=1.0) to skip the take_cover check
        berserker = TacticsEngine(personality=PERSONALITY_PRESETS["berserker"])
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=threats, allies_nearby=3, in_cover=False,
            cover_positions=[],  # No cover available
            has_los_to_threats=[True, True, True],
            squad_order=None,
        )
        action = berserker.decide_action(sit)
        assert action.action_type == "throw_grenade"

    def test_engage_default_with_threats(self, engine: TacticsEngine):
        """With threats, no cover, and no special conditions, should engage."""
        # Use berserker to skip cover-seeking
        berserker = TacticsEngine(personality=PERSONALITY_PRESETS["berserker"])
        threats = [
            ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0),
        ]
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=threats, allies_nearby=3, in_cover=False,
            cover_positions=[], has_los_to_threats=[True],
            squad_order=None,
        )
        action = berserker.decide_action(sit)
        assert action.action_type == "engage"

    def test_no_threats_no_cover_hold_or_advance(self, engine: TacticsEngine):
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=1.0, unit_ammo=1.0, unit_morale=1.0,
            threats=[], allies_nearby=0, in_cover=False,
            cover_positions=[], has_los_to_threats=[],
            squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type in ("advance", "hold")

    def test_retreat_has_position(self, engine: TacticsEngine):
        """Retreat action should provide a retreat position."""
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.05, unit_ammo=0.8, unit_morale=0.8,
            threats=[ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)],
            allies_nearby=2, in_cover=True, cover_positions=[(10, 10)],
            has_los_to_threats=[True], squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type == "retreat"
        assert action.target_pos is not None
        # Retreat should be AWAY from the threat
        threat_dist = distance((0, 0), (50, 0))
        retreat_dist = distance(action.target_pos, (50, 0))
        assert retreat_dist > threat_dist


# ---------------------------------------------------------------------------
# Reasoning text tests
# ---------------------------------------------------------------------------


class TestReasoning:
    """All actions should have human-readable reasoning."""

    def test_reasoning_is_string(self, engine: TacticsEngine):
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=[], allies_nearby=0, in_cover=False,
            cover_positions=[], has_los_to_threats=[],
            squad_order=None,
        )
        action = engine.decide_action(sit)
        assert isinstance(action.reasoning, str)
        assert len(action.reasoning) > 5

    def test_retreat_reasoning_mentions_cause(self, engine: TacticsEngine):
        # Health-based retreat
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.05, unit_ammo=0.8, unit_morale=0.8,
            threats=[ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)],
            allies_nearby=0, in_cover=False, cover_positions=[],
            has_los_to_threats=[True], squad_order=None,
        )
        action = engine.decide_action(sit)
        assert "health" in action.reasoning.lower() or "critical" in action.reasoning.lower()

    def test_engage_reasoning_mentions_target(self, engine: TacticsEngine):
        threats = [ThreatAssessment("tango_1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)]
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=threats, allies_nearby=3, in_cover=True,
            cover_positions=[(10, 10)], has_los_to_threats=[True],
            squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type == "engage"
        assert "tango_1" in action.reasoning


# ---------------------------------------------------------------------------
# Personality effects tests
# ---------------------------------------------------------------------------


class TestPersonalityEffects:
    """Tests that personality traits modify decisions appropriately."""

    def test_veteran_holds_under_fire(self, veteran_engine: TacticsEngine):
        """Veteran with high discipline should not retreat at moderate morale."""
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.4, unit_ammo=0.6, unit_morale=0.4,
            threats=[ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)],
            allies_nearby=2, in_cover=True, cover_positions=[(10, 10)],
            has_los_to_threats=[True], squad_order=None,
        )
        action = veteran_engine.decide_action(sit)
        # Veteran (discipline=0.9) should NOT retreat at morale 0.4
        assert action.action_type != "retreat"

    def test_recruit_panics_at_low_morale(self, recruit_engine: TacticsEngine):
        """Recruit with low discipline should retreat sooner."""
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.5, unit_ammo=0.6, unit_morale=0.4,
            threats=[ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)],
            allies_nearby=2, in_cover=True, cover_positions=[(10, 10)],
            has_los_to_threats=[True], squad_order=None,
        )
        action = recruit_engine.decide_action(sit)
        # Recruit (discipline=0.3) retreats at morale 0.4
        # morale_retreat_threshold = 0.3 * (2.0 - 0.3) = 0.51
        assert action.action_type == "retreat"

    def test_berserker_engages_without_cover(self, berserker_engine: TacticsEngine):
        """Berserker with max aggression should engage even without cover."""
        threats = [ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)]
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=threats, allies_nearby=3, in_cover=False,
            cover_positions=[(10, 10)], has_los_to_threats=[True],
            squad_order=None,
        )
        action = berserker_engine.decide_action(sit)
        # Berserker (aggression=1.0) skips take_cover since aggression >= 0.8
        assert action.action_type == "engage"

    def test_medic_prioritizes_healing(self, medic_engine: TacticsEngine):
        """Medic personality should choose heal_ally when conditions are met."""
        # No flanking, not outnumbered, not in cover but no cover positions,
        # aggression is very low so we go past the take_cover check
        threats = [ThreatAssessment("e1", (50, 0), 50.0, 0.3, False, False, 0.0, 1.0)]
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=threats, allies_nearby=3, in_cover=False,
            cover_positions=[], has_los_to_threats=[True],
            squad_order=None,
        )
        action = medic_engine.decide_action(sit)
        # Medic (teamwork=1.0, aggression=0.1) but aggression < 0.8 and there are cover positions...
        # Actually cover_positions is empty so skip take_cover
        assert action.action_type == "heal_ally"

    def test_sniper_cautious_no_threats(self, sniper_engine: TacticsEngine):
        """Sniper with low aggression should hold when no threats visible."""
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=[], allies_nearby=0, in_cover=True,
            cover_positions=[(10, 10)], has_los_to_threats=[],
            squad_order=None,
        )
        action = sniper_engine.decide_action(sit)
        # Sniper (aggression=0.2 < 0.5) -> hold
        assert action.action_type == "hold"

    def test_aggressive_personality_advances(self):
        """High aggression personality advances when no threats."""
        aggressive = TacticsEngine(personality=AIPersonality(aggression=0.8))
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=[], allies_nearby=2, in_cover=False,
            cover_positions=[], has_los_to_threats=[],
            squad_order=None,
        )
        action = aggressive.decide_action(sit)
        assert action.action_type == "advance"

    def test_disciplined_ignores_moderate_health_drop(self, veteran_engine: TacticsEngine):
        """Disciplined unit does not retreat at 15% health due to adjusted threshold."""
        # health_retreat_threshold = 0.2 * (2.0 - 0.9) = 0.22
        # 0.15 < 0.22 so veteran WILL retreat even with high discipline
        # Let's use health=0.25 which is > 0.22
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.25, unit_ammo=0.8, unit_morale=0.8,
            threats=[ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)],
            allies_nearby=2, in_cover=True, cover_positions=[(10, 10)],
            has_los_to_threats=[True], squad_order=None,
        )
        action = veteran_engine.decide_action(sit)
        assert action.action_type != "retreat"

    def test_undisciplined_retreats_at_higher_health(self, recruit_engine: TacticsEngine):
        """Undisciplined unit retreats at higher health threshold."""
        # health_retreat_threshold = 0.2 * (2.0 - 0.3) = 0.34
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.3, unit_ammo=0.8, unit_morale=0.8,
            threats=[ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)],
            allies_nearby=2, in_cover=True, cover_positions=[(10, 10)],
            has_los_to_threats=[True], squad_order=None,
        )
        action = recruit_engine.decide_action(sit)
        assert action.action_type == "retreat"


# ---------------------------------------------------------------------------
# Personality presets validation tests
# ---------------------------------------------------------------------------


class TestPersonalityPresets:
    """All presets should create valid AI personalities."""

    def test_all_presets_exist(self):
        expected = {"veteran", "recruit", "berserker", "sniper", "medic", "leader"}
        assert set(PERSONALITY_PRESETS.keys()) == expected

    @pytest.mark.parametrize("name", list(PERSONALITY_PRESETS.keys()))
    def test_preset_valid_ranges(self, name: str):
        p = PERSONALITY_PRESETS[name]
        assert 0.0 <= p.aggression <= 1.0
        assert 0.0 <= p.discipline <= 1.0
        assert 0.0 <= p.teamwork <= 1.0
        assert isinstance(p.accuracy_bonus, float)

    @pytest.mark.parametrize("name", list(PERSONALITY_PRESETS.keys()))
    def test_preset_creates_engine(self, name: str):
        engine = TacticsEngine(personality=PERSONALITY_PRESETS[name])
        assert engine.personality is not None

    @pytest.mark.parametrize("name", list(PERSONALITY_PRESETS.keys()))
    def test_preset_can_decide(self, name: str):
        engine = TacticsEngine(personality=PERSONALITY_PRESETS[name])
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=[], allies_nearby=0, in_cover=False,
            cover_positions=[], has_los_to_threats=[],
            squad_order=None,
        )
        action = engine.decide_action(sit)
        assert isinstance(action, TacticalAction)
        assert isinstance(action.action_type, str)
        assert isinstance(action.reasoning, str)


# ---------------------------------------------------------------------------
# Squad coordination tests
# ---------------------------------------------------------------------------


class TestSquadCoordination:
    """Tests for decide_squad_action."""

    def test_squad_returns_actions_per_unit(self, engine: TacticsEngine):
        units = [_make_unit(f"u{i}", pos=(i * 5.0, 0.0)) for i in range(4)]
        threats = [ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)]
        result = engine.decide_squad_action({
            "units": units,
            "threats": threats,
            "cover_positions": [(10, 10)],
            "objective": (100, 0),
        })
        assert len(result) == 4
        unit_ids = [uid for uid, _ in result]
        assert set(unit_ids) == {f"u{i}" for i in range(4)}

    def test_squad_fire_team_split(self, engine: TacticsEngine):
        """Squad should split into suppressors and movers."""
        units = [_make_unit(f"u{i}", pos=(i * 5.0, 0.0)) for i in range(4)]
        threats = [ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)]
        result = engine.decide_squad_action({
            "units": units,
            "threats": threats,
            "cover_positions": [],
            "objective": (100, 0),
        })
        actions = {uid: act.action_type for uid, act in result}
        action_types = set(actions.values())
        # Should have both suppress and flank
        assert "suppress" in action_types
        assert "flank" in action_types

    def test_medic_heals_wounded(self, engine: TacticsEngine):
        """Medic should prioritize healing wounded allies."""
        units = [
            _make_unit("u1", pos=(0, 0), health=0.3, role="rifleman"),
            _make_unit("u2", pos=(5, 0), health=1.0, role="rifleman"),
            _make_unit("medic1", pos=(2, 0), health=1.0, role="medic"),
        ]
        threats = [ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)]
        result = engine.decide_squad_action({
            "units": units,
            "threats": threats,
            "cover_positions": [],
            "objective": (100, 0),
        })
        medic_action = next(act for uid, act in result if uid == "medic1")
        assert medic_action.action_type == "heal_ally"
        assert medic_action.target_id == "u1"

    def test_medic_holds_when_no_wounded(self, engine: TacticsEngine):
        """Medic with no wounded allies should hold position."""
        units = [
            _make_unit("u1", pos=(0, 0), health=1.0, role="rifleman"),
            _make_unit("medic1", pos=(2, 0), health=1.0, role="medic"),
        ]
        threats = [ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)]
        result = engine.decide_squad_action({
            "units": units,
            "threats": threats,
            "cover_positions": [],
            "objective": (100, 0),
        })
        medic_action = next(act for uid, act in result if uid == "medic1")
        assert medic_action.action_type == "hold"

    def test_no_threats_everyone_advances(self, engine: TacticsEngine):
        units = [_make_unit(f"u{i}", pos=(i * 5.0, 0.0)) for i in range(3)]
        result = engine.decide_squad_action({
            "units": units,
            "threats": [],
            "cover_positions": [],
            "objective": (100, 0),
        })
        for uid, action in result:
            assert action.action_type == "advance"

    def test_empty_squad(self, engine: TacticsEngine):
        result = engine.decide_squad_action({
            "units": [],
            "threats": [],
            "cover_positions": [],
            "objective": None,
        })
        assert result == []

    def test_squad_flank_target_matches(self, engine: TacticsEngine):
        """Flanking team should target the nearest threat."""
        units = [_make_unit(f"u{i}", pos=(i * 5.0, 0.0)) for i in range(4)]
        threats = [ThreatAssessment("target_x", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)]
        result = engine.decide_squad_action({
            "units": units,
            "threats": threats,
            "cover_positions": [],
            "objective": (100, 0),
        })
        flank_actions = [act for uid, act in result if act.action_type == "flank"]
        assert len(flank_actions) > 0
        for act in flank_actions:
            assert act.target_id == "target_x"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for internal helper functions."""

    def test_centroid_single(self):
        assert _centroid([(10.0, 20.0)]) == (10.0, 20.0)

    def test_centroid_multiple(self):
        c = _centroid([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)])
        assert c[0] == pytest.approx(10.0 / 3.0)
        assert c[1] == pytest.approx(10.0 / 3.0)

    def test_centroid_empty(self):
        assert _centroid([]) == (0.0, 0.0)

    def test_cluster_radius(self):
        points = [(0.0, 0.0), (3.0, 0.0), (-3.0, 0.0)]
        center = (0.0, 0.0)
        assert _cluster_radius(points, center) == pytest.approx(3.0)

    def test_cluster_radius_empty(self):
        assert _cluster_radius([], (0.0, 0.0)) == 0.0

    def test_nearest_threat(self):
        threats = [
            ThreatAssessment("far", (100, 0), 100.0, 0.2, False, False, 0.0, 1.0),
            ThreatAssessment("near", (10, 0), 10.0, 0.8, False, False, 0.0, 1.0),
            ThreatAssessment("mid", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0),
        ]
        assert _nearest_threat(threats).threat_id == "near"

    def test_nearest_threat_empty(self):
        assert _nearest_threat([]) is None

    def test_highest_threat(self):
        threats = [
            ThreatAssessment("low", (50, 0), 50.0, 0.2, False, False, 0.0, 1.0),
            ThreatAssessment("high", (50, 0), 50.0, 0.9, False, False, 0.0, 1.0),
        ]
        assert _highest_threat(threats).threat_id == "high"

    def test_highest_threat_empty(self):
        assert _highest_threat([]) is None

    def test_threats_clustered_true(self):
        threats = [
            ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0),
            ThreatAssessment("e2", (52, 1), 52.0, 0.4, False, False, 0.0, 1.0),
        ]
        assert _threats_clustered(threats, radius=10.0)

    def test_threats_clustered_false(self):
        threats = [
            ThreatAssessment("e1", (0, 0), 50.0, 0.5, False, False, 0.0, 1.0),
            ThreatAssessment("e2", (100, 100), 140.0, 0.3, False, False, 0.0, 1.0),
        ]
        assert not _threats_clustered(threats, radius=10.0)

    def test_threats_clustered_single(self):
        threats = [ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)]
        assert not _threats_clustered(threats, radius=10.0)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_health(self, engine: TacticsEngine):
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.0, unit_ammo=1.0, unit_morale=1.0,
            threats=[ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)],
            allies_nearby=0, in_cover=False, cover_positions=[],
            has_los_to_threats=[True], squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type == "retreat"

    def test_zero_ammo(self, engine: TacticsEngine):
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=1.0, unit_ammo=0.0, unit_morale=1.0,
            threats=[ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)],
            allies_nearby=0, in_cover=True, cover_positions=[(10, 10)],
            has_los_to_threats=[True], squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type == "retreat"

    def test_zero_morale(self, engine: TacticsEngine):
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=1.0, unit_ammo=1.0, unit_morale=0.0,
            threats=[ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)],
            allies_nearby=0, in_cover=True, cover_positions=[(10, 10)],
            has_los_to_threats=[True], squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type == "retreat"

    def test_priority_is_float(self, engine: TacticsEngine):
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=[], allies_nearby=0, in_cover=False,
            cover_positions=[], has_los_to_threats=[],
            squad_order=None,
        )
        action = engine.decide_action(sit)
        assert isinstance(action.priority, float)
        assert 0.0 <= action.priority <= 1.0

    def test_action_type_is_valid(self, engine: TacticsEngine):
        valid_types = {
            "engage", "suppress", "flank", "retreat", "advance", "hold",
            "heal_ally", "throw_grenade", "take_cover", "relocate", "overwatch",
        }
        # Run through several scenarios and check action types
        scenarios = [
            TacticalSituation((0, 0), 0.1, 0.8, 0.8, [], 0, False, [], [], None),
            TacticalSituation((0, 0), 0.8, 0.8, 0.8, [], 0, False, [], [], None),
            TacticalSituation(
                (0, 0), 0.8, 0.8, 0.8,
                [ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)],
                3, True, [(10, 10)], [True], None,
            ),
        ]
        for sit in scenarios:
            action = engine.decide_action(sit)
            assert action.action_type in valid_types, f"Invalid action type: {action.action_type}"

    def test_many_threats(self, engine: TacticsEngine):
        """Engine should handle large number of threats without error."""
        enemies = [_make_enemy(f"e{i}", (30 + i, i * 2)) for i in range(50)]
        threats = engine.assess_threats((0, 0), enemies, dt=0.1)
        assert len(threats) == 50

    def test_overlapping_positions(self, engine: TacticsEngine):
        """Enemies at the same position should not crash."""
        enemies = [
            _make_enemy("e1", (50, 50), damage=5.0),
            _make_enemy("e2", (50, 50), damage=3.0),
        ]
        threats = engine.assess_threats((0, 0), enemies, dt=0.1)
        assert len(threats) == 2

    def test_in_cover_with_no_los_gives_overwatch(self, engine: TacticsEngine):
        """In cover but no LOS should give overwatch."""
        threats = [ThreatAssessment("e1", (50, 0), 50.0, 0.5, False, False, 0.0, 1.0)]
        sit = TacticalSituation(
            unit_pos=(0, 0), unit_health=0.8, unit_ammo=0.8, unit_morale=0.8,
            threats=threats, allies_nearby=3, in_cover=True,
            cover_positions=[(10, 10)], has_los_to_threats=[False],
            squad_order=None,
        )
        action = engine.decide_action(sit)
        assert action.action_type == "overwatch"
