"""Tests for the medical/casualty system.

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import random

import pytest

from tritium_lib.sim_engine.medical import (
    BODY_PARTS,
    INJURY_TABLES,
    TRIAGE_COLORS,
    CasualtyState,
    EvacRequest,
    Injury,
    InjurySeverity,
    InjuryType,
    MedicalEngine,
    TriageCategory,
    _BLEED_RATES,
    _PAIN_VALUES,
    _TREATMENT_TIMES,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_injury_types_count(self):
        assert len(InjuryType) == 7

    def test_injury_type_values(self):
        assert InjuryType.GUNSHOT.value == "gunshot"
        assert InjuryType.SHRAPNEL.value == "shrapnel"
        assert InjuryType.BURN.value == "burn"
        assert InjuryType.BLAST.value == "blast"
        assert InjuryType.CRUSH.value == "crush"
        assert InjuryType.LACERATION.value == "laceration"
        assert InjuryType.CONCUSSION.value == "concussion"

    def test_severity_count(self):
        assert len(InjurySeverity) == 5

    def test_severity_ordering(self):
        order = list(InjurySeverity)
        assert order.index(InjurySeverity.MINOR) < order.index(InjurySeverity.FATAL)

    def test_triage_categories(self):
        assert len(TriageCategory) == 4
        assert TriageCategory.IMMEDIATE.value == "immediate"
        assert TriageCategory.DELAYED.value == "delayed"
        assert TriageCategory.MINIMAL.value == "minimal"
        assert TriageCategory.EXPECTANT.value == "expectant"

    def test_triage_colors(self):
        assert TRIAGE_COLORS[TriageCategory.IMMEDIATE] == "#ff0000"
        assert TRIAGE_COLORS[TriageCategory.DELAYED] == "#ffff00"
        assert TRIAGE_COLORS[TriageCategory.MINIMAL] == "#00ff00"
        assert TRIAGE_COLORS[TriageCategory.EXPECTANT] == "#000000"


# ---------------------------------------------------------------------------
# Injury tables
# ---------------------------------------------------------------------------

class TestInjuryTables:
    def test_all_types_have_tables(self):
        for t in InjuryType:
            assert t in INJURY_TABLES, f"Missing table for {t}"

    def test_gunshot_table_sums_to_approx_one(self):
        total = sum(w for _, w in INJURY_TABLES[InjuryType.GUNSHOT])
        assert abs(total - 0.9) < 0.01  # 90% body coverage, 10% head

    def test_concussion_only_hits_head(self):
        parts = INJURY_TABLES[InjuryType.CONCUSSION]
        assert len(parts) == 1
        assert parts[0][0] == "head"
        assert parts[0][1] == 1.0

    def test_shrapnel_uniform_distribution(self):
        for _, w in INJURY_TABLES[InjuryType.SHRAPNEL]:
            assert abs(w - 1 / 6) < 0.001

    def test_bleed_rates_increase_with_severity(self):
        prev = 0.0
        for sev in InjurySeverity:
            rate = _BLEED_RATES[sev]
            assert rate >= prev
            prev = rate

    def test_pain_values_increase_with_severity(self):
        prev = 0.0
        for sev in InjurySeverity:
            p = _PAIN_VALUES[sev]
            assert p >= prev
            prev = p

    def test_treatment_times_increase_with_severity(self):
        prev = 0.0
        for sev in InjurySeverity:
            t = _TREATMENT_TIMES[sev]
            assert t >= prev
            prev = t


# ---------------------------------------------------------------------------
# Injury dataclass
# ---------------------------------------------------------------------------

class TestInjury:
    def test_create_injury(self):
        inj = Injury(
            injury_id="test1",
            injury_type=InjuryType.GUNSHOT,
            severity=InjurySeverity.SEVERE,
            body_part="torso",
            bleed_rate=0.05,
            pain=0.5,
            mobility_penalty=0.0,
            accuracy_penalty=0.0,
        )
        assert inj.injury_id == "test1"
        assert not inj.treated
        assert inj.time_since == 0.0

    def test_limb_injury(self):
        for part in ("left_arm", "right_arm", "left_leg", "right_leg"):
            inj = Injury("x", InjuryType.GUNSHOT, InjurySeverity.MINOR, part, 0, 0, 0, 0)
            assert inj.is_limb_injury()

    def test_non_limb_injury(self):
        for part in ("head", "torso"):
            inj = Injury("x", InjuryType.GUNSHOT, InjurySeverity.MINOR, part, 0, 0, 0, 0)
            assert not inj.is_limb_injury()


# ---------------------------------------------------------------------------
# CasualtyState dataclass
# ---------------------------------------------------------------------------

class TestCasualtyState:
    def test_defaults(self):
        cs = CasualtyState(unit_id="u1")
        assert cs.blood_level == 1.0
        assert cs.consciousness
        assert cs.triage == TriageCategory.MINIMAL
        assert cs.being_treated_by is None
        assert cs.evacuation_status == "none"
        assert not cs.is_dead

    def test_is_dead(self):
        cs = CasualtyState(unit_id="u1", blood_level=0.0)
        assert cs.is_dead

    def test_total_bleed_rate(self):
        inj1 = Injury("a", InjuryType.GUNSHOT, InjurySeverity.MINOR, "torso", 0.01, 0, 0, 0)
        inj2 = Injury("b", InjuryType.GUNSHOT, InjurySeverity.MINOR, "torso", 0.02, 0, 0, 0)
        cs = CasualtyState(unit_id="u1", injuries=[inj1, inj2])
        assert abs(cs.total_bleed_rate - 0.03) < 0.001

    def test_treated_injuries_dont_bleed(self):
        inj = Injury("a", InjuryType.GUNSHOT, InjurySeverity.MINOR, "torso", 0.05, 0, 0, 0, treated=True)
        cs = CasualtyState(unit_id="u1", injuries=[inj])
        assert cs.total_bleed_rate == 0.0

    def test_total_pain_capped(self):
        injuries = [
            Injury("a", InjuryType.GUNSHOT, InjurySeverity.CRITICAL, "torso", 0, 0.8, 0, 0),
            Injury("b", InjuryType.GUNSHOT, InjurySeverity.CRITICAL, "torso", 0, 0.8, 0, 0),
        ]
        cs = CasualtyState(unit_id="u1", injuries=injuries)
        assert cs.total_pain == 1.0

    def test_has_fatal_injury(self):
        inj = Injury("a", InjuryType.GUNSHOT, InjurySeverity.FATAL, "head", 0.2, 1.0, 0, 0)
        cs = CasualtyState(unit_id="u1", injuries=[inj])
        assert cs.has_fatal_injury

    def test_no_fatal_injury(self):
        inj = Injury("a", InjuryType.GUNSHOT, InjurySeverity.MINOR, "torso", 0, 0, 0, 0)
        cs = CasualtyState(unit_id="u1", injuries=[inj])
        assert not cs.has_fatal_injury

    def test_worst_severity(self):
        injuries = [
            Injury("a", InjuryType.GUNSHOT, InjurySeverity.MINOR, "torso", 0, 0, 0, 0),
            Injury("b", InjuryType.SHRAPNEL, InjurySeverity.SEVERE, "left_arm", 0, 0, 0, 0),
        ]
        cs = CasualtyState(unit_id="u1", injuries=injuries)
        assert cs.worst_severity == InjurySeverity.SEVERE

    def test_worst_severity_empty(self):
        cs = CasualtyState(unit_id="u1")
        assert cs.worst_severity is None

    def test_mobility_penalty(self):
        inj = Injury("a", InjuryType.GUNSHOT, InjurySeverity.SEVERE, "left_leg", 0, 0, 0.5, 0)
        cs = CasualtyState(unit_id="u1", injuries=[inj])
        assert cs.total_mobility_penalty == 0.5

    def test_accuracy_penalty(self):
        inj = Injury("a", InjuryType.GUNSHOT, InjurySeverity.SEVERE, "right_arm", 0, 0, 0, 0.4)
        cs = CasualtyState(unit_id="u1", injuries=[inj])
        assert cs.total_accuracy_penalty == 0.4


# ---------------------------------------------------------------------------
# MedicalEngine — inflict_injury
# ---------------------------------------------------------------------------

class TestInflictInjury:
    def test_basic_inflict(self):
        eng = MedicalEngine()
        inj = eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.MINOR)
        assert inj.injury_type == InjuryType.GUNSHOT
        assert inj.body_part == "torso"
        assert inj.severity == InjurySeverity.MINOR
        assert "u1" in eng.casualties

    def test_auto_body_part(self):
        eng = MedicalEngine()
        rng = random.Random(42)
        inj = eng.inflict_injury("u1", InjuryType.GUNSHOT, rng=rng)
        assert inj.body_part in BODY_PARTS

    def test_auto_severity(self):
        eng = MedicalEngine()
        rng = random.Random(42)
        inj = eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", rng=rng)
        assert isinstance(inj.severity, InjurySeverity)

    def test_leg_injury_has_mobility_penalty(self):
        eng = MedicalEngine()
        inj = eng.inflict_injury("u1", InjuryType.GUNSHOT, "left_leg", InjurySeverity.SEVERE)
        assert inj.mobility_penalty > 0

    def test_arm_injury_has_accuracy_penalty(self):
        eng = MedicalEngine()
        inj = eng.inflict_injury("u1", InjuryType.GUNSHOT, "right_arm", InjurySeverity.SEVERE)
        assert inj.accuracy_penalty > 0

    def test_head_injury_has_accuracy_penalty(self):
        eng = MedicalEngine()
        inj = eng.inflict_injury("u1", InjuryType.GUNSHOT, "head", InjurySeverity.SEVERE)
        assert inj.accuracy_penalty > 0

    def test_torso_has_no_mobility_or_accuracy(self):
        eng = MedicalEngine()
        inj = eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE)
        assert inj.mobility_penalty == 0.0
        assert inj.accuracy_penalty == 0.0

    def test_concussion_no_bleed(self):
        eng = MedicalEngine()
        inj = eng.inflict_injury("u1", InjuryType.CONCUSSION, "head", InjurySeverity.SEVERE)
        assert inj.bleed_rate == 0.0

    def test_gunshot_bleeds_more_than_burn(self):
        eng = MedicalEngine()
        gs = eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE)
        burn = eng.inflict_injury("u2", InjuryType.BURN, "torso", InjurySeverity.SEVERE)
        assert gs.bleed_rate > burn.bleed_rate

    def test_multiple_injuries_same_unit(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.MINOR)
        eng.inflict_injury("u1", InjuryType.SHRAPNEL, "left_arm", InjurySeverity.MODERATE)
        assert len(eng.casualties["u1"].injuries) == 2

    def test_invalid_body_part_defaults_to_torso(self):
        eng = MedicalEngine()
        inj = eng.inflict_injury("u1", InjuryType.GUNSHOT, "invalid_part", InjurySeverity.MINOR)
        assert inj.body_part == "torso"

    def test_deterministic_with_seed(self):
        eng1 = MedicalEngine()
        eng2 = MedicalEngine()
        inj1 = eng1.inflict_injury("u1", InjuryType.GUNSHOT, rng=random.Random(123))
        inj2 = eng2.inflict_injury("u1", InjuryType.GUNSHOT, rng=random.Random(123))
        assert inj1.body_part == inj2.body_part
        assert inj1.severity == inj2.severity


# ---------------------------------------------------------------------------
# MedicalEngine — blast / burn helpers
# ---------------------------------------------------------------------------

class TestBlastAndBurn:
    def test_blast_creates_concussion(self):
        eng = MedicalEngine()
        injuries = eng.inflict_blast("u1", 2.0, 10.0, rng=random.Random(42))
        assert any(i.injury_type == InjuryType.CONCUSSION for i in injuries)

    def test_blast_outside_radius_no_injury(self):
        eng = MedicalEngine()
        injuries = eng.inflict_blast("u1", 15.0, 10.0)
        assert len(injuries) == 0

    def test_close_blast_more_severe(self):
        eng = MedicalEngine()
        close = eng.inflict_blast("u1", 1.0, 10.0, rng=random.Random(42))
        eng2 = MedicalEngine()
        far = eng2.inflict_blast("u2", 9.0, 10.0, rng=random.Random(42))
        close_conc = [i for i in close if i.injury_type == InjuryType.CONCUSSION][0]
        far_conc = [i for i in far if i.injury_type == InjuryType.CONCUSSION][0]
        sev_order = list(InjurySeverity)
        assert sev_order.index(close_conc.severity) >= sev_order.index(far_conc.severity)

    def test_burn_outside_radius(self):
        eng = MedicalEngine()
        result = eng.inflict_burn("u1", 15.0, 10.0)
        assert result is None

    def test_burn_inside_radius(self):
        eng = MedicalEngine()
        inj = eng.inflict_burn("u1", 1.0, 10.0, rng=random.Random(42))
        assert inj is not None
        assert inj.injury_type == InjuryType.BURN

    def test_close_burn_more_severe(self):
        eng = MedicalEngine()
        close = eng.inflict_burn("u1", 0.5, 10.0, rng=random.Random(42))
        eng2 = MedicalEngine()
        far = eng2.inflict_burn("u2", 9.0, 10.0, rng=random.Random(42))
        sev_order = list(InjurySeverity)
        assert sev_order.index(close.severity) >= sev_order.index(far.severity)


# ---------------------------------------------------------------------------
# MedicalEngine — triage
# ---------------------------------------------------------------------------

class TestTriage:
    def test_no_injuries_minimal(self):
        eng = MedicalEngine()
        assert eng.triage("u1") == TriageCategory.MINIMAL

    def test_minor_injury_minimal(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.LACERATION, "left_arm", InjurySeverity.MINOR)
        assert eng.triage("u1") == TriageCategory.MINIMAL

    def test_moderate_injury_delayed(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.MODERATE)
        assert eng.triage("u1") == TriageCategory.DELAYED

    def test_severe_injury_immediate(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE)
        assert eng.triage("u1") == TriageCategory.IMMEDIATE

    def test_critical_injury_immediate(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.CRITICAL)
        assert eng.triage("u1") == TriageCategory.IMMEDIATE

    def test_fatal_injury_expectant(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "head", InjurySeverity.FATAL)
        assert eng.triage("u1") == TriageCategory.EXPECTANT

    def test_low_blood_immediate(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.LACERATION, "left_arm", InjurySeverity.MINOR)
        eng.casualties["u1"].blood_level = 0.35
        assert eng.triage("u1") == TriageCategory.IMMEDIATE

    def test_very_low_blood_expectant(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.LACERATION, "left_arm", InjurySeverity.MINOR)
        eng.casualties["u1"].blood_level = 0.05
        assert eng.triage("u1") == TriageCategory.EXPECTANT

    def test_dead_unit_expectant(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.MINOR)
        eng.casualties["u1"].blood_level = 0.0
        assert eng.triage("u1") == TriageCategory.EXPECTANT


# ---------------------------------------------------------------------------
# MedicalEngine — treatment
# ---------------------------------------------------------------------------

class TestTreatment:
    def test_assign_medic(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.MINOR)
        assert eng.assign_medic("m1", "u1")
        assert eng.medics["m1"] == "u1"
        assert eng.casualties["u1"].being_treated_by == "m1"

    def test_assign_medic_to_dead_fails(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.MINOR)
        eng.casualties["u1"].blood_level = 0.0
        assert not eng.assign_medic("m1", "u1")

    def test_assign_medic_no_patient_fails(self):
        eng = MedicalEngine()
        assert not eng.assign_medic("m1", "nonexistent")

    def test_treat_progress(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.LACERATION, "left_arm", InjurySeverity.MINOR)
        result = eng.treat("m1", "u1", 2.0)
        assert result["status"] == "treating"
        assert 0 < result["progress"] < 1.0

    def test_treat_completes(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.LACERATION, "left_arm", InjurySeverity.MINOR)
        # MINOR treatment time is 5s
        result = eng.treat("m1", "u1", 10.0)
        assert result["status"] == "all_treated"
        assert eng.casualties["u1"].injuries[0].treated

    def test_treat_reduces_bleed(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE)
        original_bleed = eng.casualties["u1"].injuries[0].bleed_rate
        assert original_bleed > 0
        eng.treat("m1", "u1", 100.0)  # enough to finish
        assert eng.casualties["u1"].injuries[0].bleed_rate == 0.0

    def test_treat_multiple_injuries_treats_worst_first(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.LACERATION, "left_arm", InjurySeverity.MINOR)
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE)
        result = eng.treat("m1", "u1", 1.0)
        assert result["severity"] == "severe"

    def test_release_medic(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.MINOR)
        eng.assign_medic("m1", "u1")
        eng.release_medic("m1")
        assert "m1" not in eng.medics
        assert eng.casualties["u1"].being_treated_by is None

    def test_medic_reassignment(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.MINOR)
        eng.inflict_injury("u2", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE)
        eng.assign_medic("m1", "u1")
        eng.assign_medic("m1", "u2")
        assert eng.medics["m1"] == "u2"
        assert eng.casualties["u1"].being_treated_by is None
        assert eng.casualties["u2"].being_treated_by == "m1"

    def test_treat_no_injuries_left(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.LACERATION, "left_arm", InjurySeverity.MINOR)
        eng.treat("m1", "u1", 100.0)  # finish all
        result = eng.treat("m1", "u1", 1.0)
        assert result["status"] == "all_treated"


# ---------------------------------------------------------------------------
# MedicalEngine — tick
# ---------------------------------------------------------------------------

class TestTick:
    def test_bleeding_reduces_blood(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE)
        initial = eng.casualties["u1"].blood_level
        eng.tick(10.0)
        assert eng.casualties["u1"].blood_level < initial

    def test_unconscious_at_low_blood(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.CRITICAL)
        eng.casualties["u1"].blood_level = 0.31
        events = eng.tick(1.0)  # should push below 0.3
        cs = eng.casualties["u1"]
        if cs.blood_level < 0.3:
            assert not cs.consciousness
            assert any(e["type"] == "unconscious" for e in events)

    def test_death_at_zero_blood(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.CRITICAL)
        eng.casualties["u1"].blood_level = 0.01
        # Tick enough to drain
        events = eng.tick(100.0)
        assert eng.casualties["u1"].is_dead
        assert any(e["type"] == "death" for e in events)

    def test_dead_units_skip_in_tick(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.MINOR)
        eng.casualties["u1"].blood_level = 0.0
        events = eng.tick(1.0)
        # No events for already-dead units
        assert not any(e.get("unit_id") == "u1" and e["type"] == "death" for e in events)

    def test_time_since_advances(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.MINOR)
        eng.tick(5.0)
        assert eng.casualties["u1"].injuries[0].time_since == 5.0

    def test_treated_injuries_dont_bleed(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE)
        eng.treat("m1", "u1", 100.0)  # treat fully
        bl_before = eng.casualties["u1"].blood_level
        eng.tick(10.0)
        assert eng.casualties["u1"].blood_level == bl_before

    def test_triage_updates_on_tick(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.LACERATION, "left_arm", InjurySeverity.MINOR)
        assert eng.casualties["u1"].triage == TriageCategory.MINIMAL
        eng.casualties["u1"].blood_level = 0.35
        eng.tick(0.01)
        assert eng.casualties["u1"].triage == TriageCategory.IMMEDIATE


# ---------------------------------------------------------------------------
# MedicalEngine — evacuation
# ---------------------------------------------------------------------------

class TestEvacuation:
    def test_request_evac(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE)
        req = eng.request_evac("u1", (100.0, 200.0))
        assert req is not None
        assert req.unit_id == "u1"
        assert req.evac_point == (100.0, 200.0)
        assert eng.casualties["u1"].evacuation_status == "requested"

    def test_request_evac_no_injuries(self):
        eng = MedicalEngine()
        req = eng.request_evac("u1", (0.0, 0.0))
        assert req is None

    def test_evac_priority_matches_triage(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE)
        req = eng.request_evac("u1", (0.0, 0.0))
        assert req.priority == eng.casualties["u1"].triage

    def test_update_evac_status(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.MINOR)
        eng.request_evac("u1", (0.0, 0.0))
        eng.update_evac_status("u1", "in_transit")
        assert eng.casualties["u1"].evacuation_status == "in_transit"

    def test_clear_evac(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.MINOR)
        eng.request_evac("u1", (0.0, 0.0))
        eng.clear_evac("u1")
        assert eng.casualties["u1"].evacuation_status == "evacuated"
        assert len(eng.evac_requests) == 0

    def test_evac_requests_list(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.MINOR)
        eng.inflict_injury("u2", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE)
        eng.request_evac("u1", (10.0, 10.0))
        eng.request_evac("u2", (20.0, 20.0))
        assert len(eng.evac_requests) == 2


# ---------------------------------------------------------------------------
# MedicalEngine — reporting
# ---------------------------------------------------------------------------

class TestReporting:
    def test_casualty_report_empty(self):
        eng = MedicalEngine()
        report = eng.get_casualty_report()
        assert report["total_casualties"] == 0
        assert report["total_dead"] == 0

    def test_casualty_report_by_triage(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.LACERATION, "left_arm", InjurySeverity.MINOR)
        eng.inflict_injury("u2", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE)
        eng.inflict_injury("u3", InjuryType.GUNSHOT, "head", InjurySeverity.FATAL)
        report = eng.get_casualty_report()
        assert report["total_casualties"] == 3
        assert "u1" in report["by_triage"]["minimal"]
        assert "u2" in report["by_triage"]["immediate"]
        assert "u3" in report["by_triage"]["expectant"]

    def test_casualty_report_counts_dead(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.CRITICAL)
        eng.casualties["u1"].blood_level = 0.0
        report = eng.get_casualty_report()
        assert report["total_dead"] == 1

    def test_get_unit_injuries(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE)
        eng.inflict_injury("u1", InjuryType.SHRAPNEL, "left_arm", InjurySeverity.MINOR)
        injuries = eng.get_unit_injuries("u1")
        assert len(injuries) == 2
        assert injuries[0]["type"] == "gunshot"
        assert injuries[1]["type"] == "shrapnel"

    def test_get_unit_injuries_nonexistent(self):
        eng = MedicalEngine()
        assert eng.get_unit_injuries("nobody") == []


# ---------------------------------------------------------------------------
# MedicalEngine — to_three_js
# ---------------------------------------------------------------------------

class TestThreeJS:
    def test_empty_state(self):
        eng = MedicalEngine()
        out = eng.to_three_js()
        assert out == {"casualties": [], "medics": [], "evac_requests": []}

    def test_casualty_in_output(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE)
        eng.casualties["u1"].position = (10.0, 5.0)
        out = eng.to_three_js()
        assert len(out["casualties"]) == 1
        c = out["casualties"][0]
        assert c["id"] == "u1"
        assert c["x"] == 10.0
        assert c["y"] == 5.0
        assert c["triage"] == "immediate"
        assert c["color"] == "#ff0000"
        assert c["conscious"] is True

    def test_medic_in_output(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.MINOR)
        eng.assign_medic("m1", "u1")
        out = eng.to_three_js()
        assert len(out["medics"]) == 1
        assert out["medics"][0]["id"] == "m1"
        assert out["medics"][0]["treating"] == "u1"

    def test_evac_request_in_output(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE)
        eng.request_evac("u1", (20.0, 15.0))
        out = eng.to_three_js()
        assert len(out["evac_requests"]) == 1
        ev = out["evac_requests"][0]
        assert ev["id"] == "u1"
        assert ev["x"] == 20.0
        assert ev["y"] == 15.0
        assert ev["priority"] == "immediate"

    def test_blood_level_rounded(self):
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE)
        eng.casualties["u1"].blood_level = 0.33333
        out = eng.to_three_js()
        assert out["casualties"][0]["blood_level"] == 0.33


# ---------------------------------------------------------------------------
# Integration / scenario tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_combat_scenario(self):
        """Soldier gets shot, medic treats, bleeds then stabilizes."""
        eng = MedicalEngine()
        rng = random.Random(42)

        # Soldier takes gunshot
        inj = eng.inflict_injury("soldier1", InjuryType.GUNSHOT, "torso", InjurySeverity.SEVERE, rng)
        assert eng.casualties["soldier1"].triage == TriageCategory.IMMEDIATE

        # Bleeds for 10 seconds
        events = eng.tick(10.0)
        bl = eng.casualties["soldier1"].blood_level
        assert bl < 1.0

        # Medic starts treating
        result = eng.treat("medic1", "soldier1", 30.0)  # 30s = full SEVERE treatment
        assert result["status"] in ("injury_treated", "all_treated")
        assert eng.casualties["soldier1"].injuries[0].treated

        # No more bleeding
        bl_after_treat = eng.casualties["soldier1"].blood_level
        eng.tick(10.0)
        assert eng.casualties["soldier1"].blood_level == bl_after_treat

    def test_mass_casualty_event(self):
        """Blast hits 5 soldiers, triage sorts them."""
        eng = MedicalEngine()
        rng = random.Random(99)

        for i in range(5):
            dist = 2.0 * (i + 1)
            eng.inflict_blast(f"s{i}", dist, 15.0, rng)

        report = eng.get_casualty_report()
        assert report["total_casualties"] == 5
        # At least some should be immediate (close to blast)
        assert len(report["by_triage"]["immediate"]) > 0 or len(report["by_triage"]["expectant"]) > 0

    def test_evac_workflow(self):
        """Injury -> triage -> evac request -> in transit -> evacuated."""
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "left_leg", InjurySeverity.SEVERE)
        eng.request_evac("u1", (50.0, 50.0))
        assert eng.casualties["u1"].evacuation_status == "requested"
        eng.update_evac_status("u1", "in_transit")
        assert eng.casualties["u1"].evacuation_status == "in_transit"
        eng.clear_evac("u1")
        assert eng.casualties["u1"].evacuation_status == "evacuated"

    def test_bleeding_to_death(self):
        """Untreated critical injury leads to death."""
        eng = MedicalEngine()
        eng.inflict_injury("u1", InjuryType.GUNSHOT, "torso", InjurySeverity.CRITICAL)
        # Critical gunshot: bleed_rate = 0.10 * 1.5 = 0.15 hp/s
        # At 0.15/s, blood drains in ~6.67 seconds
        all_events: list[dict] = []
        for _ in range(20):
            evts = eng.tick(1.0)
            all_events.extend(evts)
            if eng.casualties["u1"].is_dead:
                break
        assert eng.casualties["u1"].is_dead
        assert any(e["type"] == "death" for e in all_events)
        # Should have gone unconscious first
        assert any(e["type"] == "unconscious" for e in all_events)
