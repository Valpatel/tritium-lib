# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.escalation — pure escalation domain logic."""

import time
from dataclasses import dataclass

import pytest

from tritium_lib.tracking.escalation import (
    THREAT_LEVELS,
    ClassifyResult,
    EscalationConfig,
    ThreatRecord,
    classify_all_targets,
    classify_target,
    escalation_index,
    find_zone,
    is_escalation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_zone(
    x: float = 0.0,
    y: float = 0.0,
    radius: float = 10.0,
    zone_type: str = "perimeter",
    name: str = "zone-A",
) -> dict:
    return {
        "position": {"x": x, "z": y},
        "type": zone_type,
        "name": name,
        "properties": {"radius": radius},
    }


@dataclass
class _FakeTarget:
    target_id: str
    alliance: str = "unknown"
    position: tuple = (0.0, 0.0)


# ---------------------------------------------------------------------------
# escalation_index / is_escalation
# ---------------------------------------------------------------------------

class TestEscalationIndex:
    def test_none(self):
        assert escalation_index("none") == 0

    def test_hostile(self):
        assert escalation_index("hostile") == 3

    def test_all_levels_ascending(self):
        indices = [escalation_index(lv) for lv in THREAT_LEVELS]
        assert indices == [0, 1, 2, 3]

    def test_invalid_returns_zero(self):
        assert escalation_index("banana") == 0

    def test_is_escalation_up(self):
        assert is_escalation("none", "hostile") is True
        assert is_escalation("unknown", "suspicious") is True

    def test_is_escalation_down(self):
        assert is_escalation("hostile", "none") is False
        assert is_escalation("suspicious", "unknown") is False

    def test_is_escalation_same(self):
        assert is_escalation("suspicious", "suspicious") is False


# ---------------------------------------------------------------------------
# find_zone
# ---------------------------------------------------------------------------

class TestFindZone:
    def test_inside_single_zone(self):
        zones = [_make_zone(x=0, y=0, radius=20)]
        result = find_zone((5.0, 5.0), zones)
        assert result is not None
        assert result["name"] == "zone-A"

    def test_outside_all_zones(self):
        zones = [_make_zone(x=0, y=0, radius=5)]
        result = find_zone((100.0, 100.0), zones)
        assert result is None

    def test_restricted_wins(self):
        zones = [
            _make_zone(x=0, y=0, radius=20, zone_type="perimeter", name="outer"),
            _make_zone(x=0, y=0, radius=20, zone_type="restricted_area", name="inner"),
        ]
        result = find_zone((5.0, 5.0), zones)
        assert result is not None
        assert result["name"] == "inner"

    def test_empty_zones(self):
        assert find_zone((0.0, 0.0), []) is None

    def test_exact_boundary(self):
        zones = [_make_zone(x=0, y=0, radius=10)]
        assert find_zone((10.0, 0.0), zones) is not None
        assert find_zone((10.1, 0.0), zones) is None

    def test_zone_with_y_key(self):
        """Zone position may use 'y' instead of 'z'."""
        zone = {
            "position": {"x": 5.0, "y": 5.0},
            "type": "perimeter",
            "name": "y-zone",
            "properties": {"radius": 10.0},
        }
        assert find_zone((5.0, 5.0), [zone]) is not None


# ---------------------------------------------------------------------------
# classify_target — single target per-tick
# ---------------------------------------------------------------------------

class TestClassifyTarget:
    def test_zone_entry_escalates_to_unknown(self):
        record = ThreatRecord(target_id="t1")
        zone = _make_zone(zone_type="perimeter")
        result, _ = classify_target(record, zone, time.monotonic())
        assert result.level_changed is True
        assert result.new_level == "unknown"

    def test_restricted_zone_escalates_to_suspicious(self):
        record = ThreatRecord(target_id="t1")
        zone = _make_zone(zone_type="restricted_area")
        result, _ = classify_target(record, zone, time.monotonic())
        assert result.level_changed is True
        assert result.new_level == "suspicious"

    def test_linger_escalates_to_hostile(self):
        config = EscalationConfig(linger_threshold=1.0)
        record = ThreatRecord(target_id="t1")
        now = time.monotonic()

        # First tick: enters zone -> unknown
        zone = _make_zone(zone_type="perimeter")
        result, exit_t = classify_target(record, zone, now, config)
        assert result.new_level == "unknown"

        # Second tick: after linger threshold -> hostile
        result, exit_t = classify_target(record, zone, now + 1.5, config, exit_t)
        assert result.level_changed is True
        assert result.new_level == "hostile"
        assert record.prior_hostile is True

    def test_de_escalation_outside_zones(self):
        config = EscalationConfig(deescalation_time=1.0)
        # Target was previously in a zone (in_zone is set)
        record = ThreatRecord(target_id="t1", threat_level="suspicious", in_zone="zone-A")
        now = time.monotonic()

        # Target leaves zone — exit time gets set
        result, exit_t = classify_target(record, None, now, config, 0.0)
        assert exit_t > 0

        # After deescalation time
        result, exit_t = classify_target(record, None, now + 1.5, config, exit_t)
        assert result.level_changed is True
        assert result.new_level == "unknown"

    def test_no_zone_no_change(self):
        record = ThreatRecord(target_id="t1")
        result, exit_t = classify_target(record, None, time.monotonic())
        assert result.level_changed is False
        assert result.new_level == "none"

    def test_prior_hostile_skips_unknown(self):
        record = ThreatRecord(target_id="t1", prior_hostile=True)
        zone = _make_zone(zone_type="perimeter")
        result, _ = classify_target(record, zone, time.monotonic())
        assert result.new_level == "suspicious"

    def test_result_reason_for_zone(self):
        record = ThreatRecord(target_id="t1")
        zone = _make_zone(zone_type="perimeter", name="north-fence")
        result, _ = classify_target(record, zone, time.monotonic())
        assert "zone:north-fence" in result.reason

    def test_result_reason_for_deescalation(self):
        config = EscalationConfig(deescalation_time=0.0)
        # Target was previously in a zone
        record = ThreatRecord(target_id="t1", threat_level="unknown", in_zone="zone-A")
        now = time.monotonic()
        # Leave zone — sets exit_t
        result, exit_t = classify_target(record, None, now, config, 0.0)
        assert exit_t > 0
        # Immediate de-escalation (deescalation_time=0)
        result, exit_t = classify_target(record, None, now + 0.01, config, exit_t)
        assert result.reason == "de-escalation"

    def test_zone_entered_field_set(self):
        record = ThreatRecord(target_id="t1")
        zone = _make_zone(name="alpha")
        result, _ = classify_target(record, zone, time.monotonic())
        assert result.zone_entered == "alpha"

    def test_zone_reentry_not_flagged(self):
        """If already in the same zone, zone_entered should be empty."""
        record = ThreatRecord(target_id="t1", in_zone="alpha")
        zone = _make_zone(name="alpha")
        result, _ = classify_target(record, zone, time.monotonic())
        assert result.zone_entered == ""


# ---------------------------------------------------------------------------
# classify_all_targets — batch processing
# ---------------------------------------------------------------------------

class TestClassifyAllTargets:
    def test_batch_creates_records(self):
        records: dict[str, ThreatRecord] = {}
        exit_times: dict[str, float] = {}
        targets = [_FakeTarget("t1"), _FakeTarget("t2")]
        zones = [_make_zone(x=0, y=0, radius=100)]

        results = classify_all_targets(
            records, exit_times, targets, zones, time.monotonic()
        )
        assert "t1" in records
        assert "t2" in records
        assert len(results) == 2  # both escalated

    def test_batch_skips_friendly(self):
        records: dict[str, ThreatRecord] = {}
        exit_times: dict[str, float] = {}
        targets = [
            _FakeTarget("f1", alliance="friendly"),
            _FakeTarget("t1", alliance="unknown"),
        ]
        zones = [_make_zone(x=0, y=0, radius=100)]

        results = classify_all_targets(
            records, exit_times, targets, zones, time.monotonic()
        )
        assert "f1" not in records
        assert "t1" in records

    def test_batch_skips_neutral(self):
        records: dict[str, ThreatRecord] = {}
        exit_times: dict[str, float] = {}
        targets = [_FakeTarget("n1", alliance="neutral")]
        zones = [_make_zone(x=0, y=0, radius=100)]

        classify_all_targets(records, exit_times, targets, zones, time.monotonic())
        assert "n1" not in records

    def test_batch_prunes_stale(self):
        records = {"gone": ThreatRecord(target_id="gone")}
        exit_times = {"gone": 1.0}
        targets = [_FakeTarget("t1")]
        zones = []

        classify_all_targets(records, exit_times, targets, zones, time.monotonic())
        assert "gone" not in records
        assert "gone" not in exit_times

    def test_batch_returns_only_changed(self):
        records: dict[str, ThreatRecord] = {}
        exit_times: dict[str, float] = {}
        targets = [_FakeTarget("t1", position=(500.0, 500.0))]  # far from zone
        zones = [_make_zone(x=0, y=0, radius=10)]

        results = classify_all_targets(
            records, exit_times, targets, zones, time.monotonic()
        )
        assert len(results) == 0  # no zone, no change

    def test_batch_with_dict_targets(self):
        """classify_all_targets should accept dict-based targets."""
        records: dict[str, ThreatRecord] = {}
        exit_times: dict[str, float] = {}
        targets = [
            {"target_id": "d1", "alliance": "unknown", "position": (5.0, 5.0)},
        ]
        zones = [_make_zone(x=0, y=0, radius=100)]

        results = classify_all_targets(
            records, exit_times, targets, zones, time.monotonic()
        )
        assert "d1" in records
        assert len(results) == 1

    def test_batch_with_dict_position_as_dict(self):
        """Position can be a dict with x/y keys."""
        records: dict[str, ThreatRecord] = {}
        exit_times: dict[str, float] = {}
        targets = [
            {"target_id": "d2", "alliance": "hostile", "position": {"x": 1.0, "y": 2.0}},
        ]
        zones = [_make_zone(x=0, y=0, radius=100)]

        results = classify_all_targets(
            records, exit_times, targets, zones, time.monotonic()
        )
        assert "d2" in records


# ---------------------------------------------------------------------------
# ThreatRecord
# ---------------------------------------------------------------------------

class TestThreatRecord:
    def test_defaults(self):
        r = ThreatRecord(target_id="abc")
        assert r.threat_level == "none"
        assert r.prior_hostile is False
        assert r.in_zone == ""

    def test_to_dict(self):
        r = ThreatRecord(target_id="abc", threat_level="hostile", prior_hostile=True)
        d = r.to_dict()
        assert d["target_id"] == "abc"
        assert d["threat_level"] == "hostile"
        assert d["prior_hostile"] is True
        assert "level_since" in d
        assert "last_update" in d


# ---------------------------------------------------------------------------
# EscalationConfig
# ---------------------------------------------------------------------------

class TestEscalationConfig:
    def test_defaults(self):
        cfg = EscalationConfig()
        assert cfg.linger_threshold == 30.0
        assert cfg.deescalation_time == 30.0
        assert cfg.tick_interval == 0.5
        # Passive decay: 60 s by default — Gap-fix C M-5
        assert cfg.passive_decay_interval == 60.0

    def test_custom(self):
        cfg = EscalationConfig(linger_threshold=5.0, deescalation_time=10.0)
        assert cfg.linger_threshold == 5.0
        assert cfg.deescalation_time == 10.0


# ---------------------------------------------------------------------------
# Passive decay (Gap-fix C M-5)
# ---------------------------------------------------------------------------

class TestPassiveDecay:
    """A target pinned at hostile inside a zone must still decay over time."""

    def test_passive_decay_in_zone_drops_one_band(self):
        # 1 s passive decay so the test runs fast.
        cfg = EscalationConfig(
            linger_threshold=999.0,
            passive_decay_interval=1.0,
        )
        record = ThreatRecord(
            target_id="t1",
            threat_level="hostile",
            in_zone="zone-A",
        )
        # level_since was set on creation; advance time past the
        # passive-decay interval but not past the linger threshold.
        now = record.level_since + 2.0
        zone = _make_zone(zone_type="restricted_area", name="zone-A")

        result, _ = classify_target(record, zone, now, cfg, 0.0)

        assert result.level_changed is True
        assert result.old_level == "hostile"
        assert result.new_level == "suspicious"
        assert "passive-decay" in result.reason

    def test_passive_decay_disabled_holds_level(self):
        cfg = EscalationConfig(passive_decay_interval=0.0)
        record = ThreatRecord(
            target_id="t1",
            threat_level="hostile",
            in_zone="zone-A",
        )
        now = record.level_since + 9999.0
        zone = _make_zone(zone_type="restricted_area", name="zone-A")

        result, _ = classify_target(record, zone, now, cfg, 0.0)

        # Without passive decay, hostile in restricted zone stays hostile.
        assert result.level_changed is False
        assert result.new_level == "hostile"

    def test_passive_decay_does_not_fire_when_just_escalated(self):
        # A fresh escalation must not be immediately decayed in the same
        # tick.  level_since gets bumped when classify_target raises the
        # level, so decay_eligible should be False on that tick.
        cfg = EscalationConfig(
            linger_threshold=999.0,
            passive_decay_interval=0.0001,  # tiny — would otherwise fire
        )
        record = ThreatRecord(target_id="t1")
        zone = _make_zone(zone_type="restricted_area", name="zone-A")
        now = time.monotonic()

        result, _ = classify_target(record, zone, now + 1.0, cfg, 0.0)

        # Restricted-zone entry from "none" raises level to "suspicious"
        # this tick.  The tiny passive_decay_interval would normally
        # trigger immediate decay, but the guard
        # `record.threat_level == old_level` blocks decay on the same
        # tick the level was raised — so we must still see "suspicious".
        assert result.new_level == "suspicious"
        assert result.level_changed is True
        assert result.old_level == "none"

    def test_passive_decay_steps_one_band_at_a_time(self):
        """Repeated ticks step down through unknown -> none."""
        cfg = EscalationConfig(passive_decay_interval=1.0)
        record = ThreatRecord(
            target_id="t1",
            threat_level="suspicious",
            in_zone="",  # outside any zone, no zone-exit timer either
        )
        # First decay: suspicious -> unknown.
        result, _ = classify_target(
            record, None, record.level_since + 2.0, cfg, 0.0,
        )
        assert result.new_level == "unknown"

        # Second decay (after another interval): unknown -> none.
        result, _ = classify_target(
            record, None, record.level_since + 2.0, cfg, 0.0,
        )
        assert result.new_level == "none"

        # Third decay: stays at none (idx 0 floor).
        result, _ = classify_target(
            record, None, record.level_since + 2.0, cfg, 0.0,
        )
        assert result.new_level == "none"
        assert result.level_changed is False
