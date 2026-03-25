# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.person_reid."""

import time

import pytest

pytestmark = pytest.mark.unit

from tritium_lib.tracking.target_tracker import TargetTracker, TrackedTarget
from tritium_lib.tracking.dossier import DossierStore
from tritium_lib.tracking.person_reid import (
    ReIDEngine,
    PersonProfile,
    MatchResult,
    MergeRecord,
    STRATEGY_WEIGHTS,
    MAX_TEMPORAL_GAP,
    MAX_WALKING_SPEED,
    DEFAULT_MATCH_THRESHOLD,
    RSSI_TOLERANCE,
    ADV_INTERVAL_TOLERANCE,
    _score_ble_mac_rotation,
    _score_temporal_cooccurrence,
    _score_spatial_consistency,
    _score_signal_fingerprint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(
    target_id: str,
    source: str = "ble",
    position: tuple[float, float] = (0.0, 0.0),
    name: str = "",
    asset_type: str = "person",
    confidence: float = 0.8,
    first_seen: float | None = None,
    last_seen: float | None = None,
    signal_count: int = 10,
) -> TrackedTarget:
    now = time.monotonic()
    return TrackedTarget(
        target_id=target_id,
        name=name or target_id,
        alliance="unknown",
        asset_type=asset_type,
        position=position,
        source=source,
        position_confidence=confidence,
        last_seen=last_seen if last_seen is not None else now,
        first_seen=first_seen if first_seen is not None else now - 30.0,
        signal_count=signal_count,
        confirming_sources={source},
        classification=asset_type,
    )


def _make_profile(
    target_id: str = "ble_aabbccddeeff",
    source: str = "ble",
    position: tuple[float, float] = (0.0, 0.0),
    first_seen: float | None = None,
    last_seen: float | None = None,
    rssi_mean: float = -60.0,
    rssi_std: float = 5.0,
    oui_prefix: str = "aabbcc",
    probe_ssids: set[str] | None = None,
    signal_count: int = 10,
    adv_interval_ms: float = 100.0,
    speed: float = 0.0,
    heading: float = 0.0,
) -> PersonProfile:
    now = time.monotonic()
    return PersonProfile(
        target_id=target_id,
        source=source,
        asset_type="person",
        position=position,
        first_seen=first_seen if first_seen is not None else now - 60.0,
        last_seen=last_seen if last_seen is not None else now,
        rssi_mean=rssi_mean,
        rssi_std=rssi_std,
        oui_prefix=oui_prefix,
        probe_ssids=probe_ssids or set(),
        signal_count=signal_count,
        adv_interval_ms=adv_interval_ms,
        speed=speed,
        heading=heading,
    )


def _engine_with_targets(*targets: TrackedTarget) -> tuple[ReIDEngine, TargetTracker]:
    """Create a ReIDEngine with pre-loaded targets."""
    tracker = TargetTracker()
    for t in targets:
        tracker._targets[t.target_id] = t
    engine = ReIDEngine(tracker=tracker)
    return engine, tracker


# ---------------------------------------------------------------------------
# PersonProfile tests
# ---------------------------------------------------------------------------

class TestPersonProfile:
    def test_create_empty(self):
        p = PersonProfile(target_id="test_1")
        assert p.target_id == "test_1"
        assert p.source == ""
        assert p.position == (0.0, 0.0)
        assert p.probe_ssids == set()

    def test_to_dict(self):
        p = _make_profile(probe_ssids={"MyWiFi", "Office"})
        d = p.to_dict()
        assert d["target_id"] == "ble_aabbccddeeff"
        assert d["source"] == "ble"
        assert "position" in d
        assert "MyWiFi" in d["probe_ssids"]
        assert "Office" in d["probe_ssids"]
        assert isinstance(d["confirming_sources"], list)

    def test_correlated_ids_default_empty(self):
        p = PersonProfile(target_id="x")
        assert p.correlated_ids == []


# ---------------------------------------------------------------------------
# MatchResult tests
# ---------------------------------------------------------------------------

class TestMatchResult:
    def test_basic(self):
        m = MatchResult(
            target_id="ble_112233",
            score=0.85,
            strategy_scores={"spatial": 0.9, "temporal": 0.7},
            detail="test",
        )
        assert m.score == 0.85
        assert m.target_id == "ble_112233"

    def test_to_dict(self):
        m = MatchResult(target_id="x", score=0.123456789)
        d = m.to_dict()
        assert d["score"] == 0.1235  # rounded to 4 places


# ---------------------------------------------------------------------------
# BLE MAC Rotation scoring
# ---------------------------------------------------------------------------

class TestBLEMACRotation:
    def test_non_ble_returns_zero(self):
        q = _make_profile(source="yolo")
        c = _make_profile(source="ble", target_id="ble_other")
        assert _score_ble_mac_rotation(q, c) == 0.0

    def test_both_non_ble_returns_zero(self):
        q = _make_profile(source="yolo")
        c = _make_profile(source="yolo", target_id="det_1")
        assert _score_ble_mac_rotation(q, c) == 0.0

    def test_same_oui_similar_rssi_high_score(self):
        now = time.monotonic()
        q = _make_profile(
            oui_prefix="aabbcc",
            rssi_mean=-60.0,
            last_seen=now - 5.0,
        )
        c = _make_profile(
            target_id="ble_aabbccddeef0",
            oui_prefix="aabbcc",
            rssi_mean=-62.0,
            first_seen=now - 3.0,
            last_seen=now,
        )
        score = _score_ble_mac_rotation(q, c)
        assert score > 0.5, f"Expected high score, got {score}"

    def test_different_oui_penalizes(self):
        now = time.monotonic()
        q = _make_profile(oui_prefix="aabbcc", rssi_mean=-60.0, last_seen=now - 5.0)
        c = _make_profile(
            target_id="ble_ddeeff112233",
            oui_prefix="ddeeff",
            rssi_mean=-62.0,
            first_seen=now - 3.0,
        )
        score = _score_ble_mac_rotation(q, c)
        # Different OUI but similar RSSI and timing — moderate score
        assert score < 0.8

    def test_both_random_macs_partial_score(self):
        now = time.monotonic()
        q = _make_profile(oui_prefix="", rssi_mean=-70.0, last_seen=now - 5.0)
        c = _make_profile(
            target_id="ble_other",
            oui_prefix="",
            rssi_mean=-72.0,
            first_seen=now - 3.0,
        )
        score = _score_ble_mac_rotation(q, c)
        assert score > 0.0, "Both random MACs should give partial score"

    def test_very_different_rssi_low_score(self):
        now = time.monotonic()
        q = _make_profile(oui_prefix="aabbcc", rssi_mean=-40.0, last_seen=now - 5.0)
        c = _make_profile(
            target_id="ble_aabbccddeef0",
            oui_prefix="aabbcc",
            rssi_mean=-95.0,
            first_seen=now - 3.0,
        )
        score = _score_ble_mac_rotation(q, c)
        # OUI match + temporal closeness boost the score even with bad RSSI
        # but it should be lower than same-RSSI cases
        assert score < 0.8, f"Expected reduced score for distant RSSI, got {score}"


# ---------------------------------------------------------------------------
# Temporal co-occurrence scoring
# ---------------------------------------------------------------------------

class TestTemporalCooccurrence:
    def test_sequential_disappear_appear(self):
        now = time.monotonic()
        q = _make_profile(first_seen=now - 120.0, last_seen=now - 10.0)
        c = _make_profile(
            target_id="ble_new",
            first_seen=now - 5.0,
            last_seen=now,
        )
        score = _score_temporal_cooccurrence(q, c)
        assert score > 0.5, f"Sequential appear/disappear should score high, got {score}"

    def test_large_gap_returns_zero(self):
        now = time.monotonic()
        q = _make_profile(last_seen=now - 300.0)
        c = _make_profile(
            target_id="ble_new",
            first_seen=now - 100.0,
        )
        score = _score_temporal_cooccurrence(q, c)
        assert score == 0.0, "Large gap should return zero"

    def test_overlapping_targets_low_score(self):
        now = time.monotonic()
        q = _make_profile(first_seen=now - 100.0, last_seen=now)
        c = _make_profile(
            target_id="ble_other",
            first_seen=now - 100.0,
            last_seen=now,
        )
        score = _score_temporal_cooccurrence(q, c)
        assert score == 0.0, "Fully overlapping targets should score zero"

    def test_near_zero_gap_high_score(self):
        now = time.monotonic()
        q = _make_profile(first_seen=now - 100.0, last_seen=now - 1.0)
        c = _make_profile(
            target_id="ble_new",
            first_seen=now - 0.5,
            last_seen=now,
        )
        score = _score_temporal_cooccurrence(q, c)
        assert score > 0.9, f"Near-zero gap should score very high, got {score}"


# ---------------------------------------------------------------------------
# Spatial consistency scoring
# ---------------------------------------------------------------------------

class TestSpatialConsistency:
    def test_same_position_high_score(self):
        now = time.monotonic()
        q = _make_profile(position=(10.0, 20.0), last_seen=now - 5.0)
        c = _make_profile(
            target_id="ble_new",
            position=(10.0, 20.0),
            first_seen=now - 3.0,
        )
        score = _score_spatial_consistency(q, c)
        assert score > 0.9, f"Same position should score high, got {score}"

    def test_walking_distance_moderate_score(self):
        now = time.monotonic()
        q = _make_profile(position=(0.0, 0.0), last_seen=now - 10.0)
        # Walking speed ~1.8 m/s, 10s gap -> max 18m
        c = _make_profile(
            target_id="ble_new",
            position=(10.0, 0.0),
            first_seen=now,
        )
        score = _score_spatial_consistency(q, c)
        assert 0.2 < score < 0.9, f"Walking distance should give moderate score, got {score}"

    def test_impossible_distance_zero(self):
        now = time.monotonic()
        q = _make_profile(position=(0.0, 0.0), last_seen=now - 2.0)
        # 2 seconds at max walking 1.8 m/s = 3.6m. 100m is way beyond.
        c = _make_profile(
            target_id="ble_new",
            position=(100.0, 0.0),
            first_seen=now,
        )
        score = _score_spatial_consistency(q, c)
        assert score == 0.0, f"Impossible distance should score zero, got {score}"

    def test_slight_overshoot_reduced(self):
        now = time.monotonic()
        q = _make_profile(position=(0.0, 0.0), last_seen=now - 10.0)
        # 10s at 1.8 m/s = 18m max. 25m is 1.39x overshoot
        c = _make_profile(
            target_id="ble_new",
            position=(25.0, 0.0),
            first_seen=now,
        )
        score = _score_spatial_consistency(q, c)
        assert 0.0 < score < 0.3, f"Slight overshoot should reduce score, got {score}"


# ---------------------------------------------------------------------------
# Signal fingerprint scoring
# ---------------------------------------------------------------------------

class TestSignalFingerprint:
    def test_identical_fingerprint(self):
        q = _make_profile(rssi_mean=-60.0, rssi_std=5.0, signal_count=100)
        c = _make_profile(
            target_id="ble_new",
            rssi_mean=-60.0,
            rssi_std=5.0,
            signal_count=100,
        )
        score = _score_signal_fingerprint(q, c)
        assert score > 0.9, f"Identical fingerprint should score high, got {score}"

    def test_different_rssi(self):
        q = _make_profile(rssi_mean=-40.0, rssi_std=3.0)
        c = _make_profile(target_id="new", rssi_mean=-90.0, rssi_std=15.0)
        score = _score_signal_fingerprint(q, c)
        # Signal rate similarity still contributes even when RSSI diverges
        assert score < 0.5, f"Very different signals should score low, got {score}"

    def test_probe_ssid_overlap(self):
        q = _make_profile(probe_ssids={"HomeNet", "OfficeWiFi", "CafeSpot"})
        c = _make_profile(
            target_id="new",
            probe_ssids={"HomeNet", "OfficeWiFi", "GymWiFi"},
        )
        score = _score_signal_fingerprint(q, c)
        # Jaccard = 2/4 = 0.5 for SSIDs, plus RSSI match
        assert score > 0.3

    def test_no_probe_ssids(self):
        q = _make_profile(probe_ssids=set())
        c = _make_profile(target_id="new", probe_ssids=set())
        # Should still score based on RSSI and signal rate
        score = _score_signal_fingerprint(q, c)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# ReIDEngine — create_profile
# ---------------------------------------------------------------------------

class TestCreateProfile:
    def test_profile_from_live_target(self):
        t = _make_target("ble_aabbccddeeff", source="ble", position=(5.0, 10.0))
        engine, tracker = _engine_with_targets(t)
        profile = engine.create_profile("ble_aabbccddeeff")
        assert profile is not None
        assert profile.target_id == "ble_aabbccddeeff"
        assert profile.source == "ble"
        assert profile.position == (5.0, 10.0)

    def test_profile_nonexistent_returns_none(self):
        engine, _ = _engine_with_targets()
        profile = engine.create_profile("doesnt_exist")
        assert profile is None

    def test_ble_oui_extraction(self):
        # Non-random MAC: first byte even -> globally unique
        t = _make_target("ble_001122334455", source="ble")
        engine, _ = _engine_with_targets(t)
        profile = engine.create_profile("ble_001122334455")
        assert profile is not None
        assert profile.oui_prefix == "001122"

    def test_random_mac_oui_empty(self):
        # Random MAC: locally administered bit set (0x02)
        # 0x02 in first byte -> random MAC
        t = _make_target("ble_02aabbccdde0", source="ble")
        engine, _ = _engine_with_targets(t)
        profile = engine.create_profile("ble_02aabbccdde0")
        assert profile is not None
        assert profile.oui_prefix == "", "Random MAC should have empty OUI"


# ---------------------------------------------------------------------------
# ReIDEngine — find_matches
# ---------------------------------------------------------------------------

class TestFindMatches:
    def test_match_similar_profiles(self):
        now = time.monotonic()
        engine, tracker = _engine_with_targets()
        query = _make_profile(
            target_id="ble_aabbccddeeff",
            position=(5.0, 5.0),
            rssi_mean=-60.0,
            last_seen=now - 5.0,
            first_seen=now - 60.0,
        )
        candidate = _make_profile(
            target_id="ble_aabbccddeef0",
            position=(6.0, 5.5),
            rssi_mean=-62.0,
            first_seen=now - 3.0,
            last_seen=now,
            oui_prefix="aabbcc",
        )
        matches = engine.find_matches(query, [candidate], threshold=0.1)
        assert len(matches) > 0
        assert matches[0].target_id == "ble_aabbccddeef0"
        assert matches[0].score > 0.0

    def test_no_self_match(self):
        engine, _ = _engine_with_targets()
        query = _make_profile(target_id="ble_aabb")
        same = _make_profile(target_id="ble_aabb")
        matches = engine.find_matches(query, [same], threshold=0.0)
        assert len(matches) == 0, "Should not match against self"

    def test_skip_already_correlated(self):
        engine, _ = _engine_with_targets()
        query = _make_profile(target_id="ble_aa")
        query.correlated_ids = ["ble_bb"]
        candidate = _make_profile(target_id="ble_bb")
        matches = engine.find_matches(query, [candidate], threshold=0.0)
        assert len(matches) == 0, "Should skip already correlated IDs"

    def test_matches_sorted_by_score(self):
        now = time.monotonic()
        engine, _ = _engine_with_targets()
        query = _make_profile(
            position=(0.0, 0.0),
            rssi_mean=-60.0,
            last_seen=now - 5.0,
        )
        # Close candidate
        c1 = _make_profile(
            target_id="ble_close",
            position=(1.0, 0.0),
            rssi_mean=-61.0,
            first_seen=now - 3.0,
            last_seen=now,
        )
        # Far candidate
        c2 = _make_profile(
            target_id="ble_far",
            position=(50.0, 50.0),
            rssi_mean=-90.0,
            first_seen=now - 3.0,
            last_seen=now,
        )
        matches = engine.find_matches(query, [c1, c2], threshold=0.0)
        if len(matches) >= 2:
            assert matches[0].score >= matches[1].score

    def test_limit_respected(self):
        engine, _ = _engine_with_targets()
        query = _make_profile()
        candidates = [
            _make_profile(target_id=f"ble_{i:012x}") for i in range(20)
        ]
        matches = engine.find_matches(query, candidates, threshold=0.0, limit=3)
        assert len(matches) <= 3


# ---------------------------------------------------------------------------
# ReIDEngine — merge_identities
# ---------------------------------------------------------------------------

class TestMergeIdentities:
    def test_merge_two_live_targets(self):
        t_a = _make_target("ble_aa", source="ble", position=(1.0, 2.0))
        t_b = _make_target("ble_bb", source="ble", position=(1.5, 2.5))
        engine, tracker = _engine_with_targets(t_a, t_b)

        record = engine.merge_identities("ble_aa", "ble_bb", score=0.8)

        assert record.primary_id == "ble_aa"
        assert record.secondary_id == "ble_bb"
        assert record.score == 0.8
        assert record.dossier_uuid != ""

        # Secondary should be removed from tracker
        assert tracker.get_target("ble_bb") is None
        # Primary should absorb secondary's ID
        primary = tracker.get_target("ble_aa")
        assert primary is not None
        assert "ble_bb" in primary.correlated_ids

    def test_merge_creates_dossier(self):
        t_a = _make_target("ble_aa", source="ble")
        t_b = _make_target("det_person_1", source="yolo")
        engine, tracker = _engine_with_targets(t_a, t_b)

        record = engine.merge_identities("ble_aa", "det_person_1", score=0.7)

        # Should be findable in dossier store
        dossier = engine.dossier_store.find_by_signal("ble_aa")
        assert dossier is not None
        assert dossier.has_signal("det_person_1")

    def test_merge_records_history(self):
        t_a = _make_target("ble_aa", source="ble")
        t_b = _make_target("ble_bb", source="ble")
        engine, _ = _engine_with_targets(t_a, t_b)

        engine.merge_identities("ble_aa", "ble_bb", score=0.75)

        history = engine.get_merge_history()
        assert len(history) == 1
        assert history[0]["primary_id"] == "ble_aa"
        assert history[0]["secondary_id"] == "ble_bb"

    def test_merge_when_one_target_missing(self):
        t_a = _make_target("ble_aa", source="ble")
        engine, tracker = _engine_with_targets(t_a)
        # ble_bb doesn't exist in tracker
        record = engine.merge_identities("ble_aa", "ble_bb", score=0.6)
        # Should still create a dossier
        assert record.dossier_uuid != ""


# ---------------------------------------------------------------------------
# ReIDEngine — record_departure and departed profiles
# ---------------------------------------------------------------------------

class TestDepartedProfiles:
    def test_record_and_retrieve(self):
        t = _make_target("ble_aabb", source="ble", position=(3.0, 4.0))
        engine, tracker = _engine_with_targets(t)

        engine.record_departure(t)
        tracker.remove("ble_aabb")

        assert engine.get_departed_count() == 1

        # Should be retrievable via create_profile fallback
        profile = engine.create_profile("ble_aabb")
        assert profile is not None
        assert profile.target_id == "ble_aabb"

    def test_departed_included_in_candidates(self):
        t = _make_target("ble_departed", source="ble")
        engine, tracker = _engine_with_targets(t)
        engine.record_departure(t)
        tracker.remove("ble_departed")

        candidates = engine.get_candidate_profiles(
            include_active=False, include_departed=True
        )
        ids = [c.target_id for c in candidates]
        assert "ble_departed" in ids

    def test_departed_eviction(self):
        engine, tracker = _engine_with_targets()
        engine._max_departed = 3

        for i in range(5):
            t = _make_target(f"ble_{i:04x}", source="ble")
            t.last_seen = time.monotonic() - (100 - i)  # newer ones have later last_seen
            engine.record_departure(t)

        assert engine.get_departed_count() <= 3


# ---------------------------------------------------------------------------
# ReIDEngine — scan_for_matches
# ---------------------------------------------------------------------------

class TestScanForMatches:
    def test_scan_empty_tracker(self):
        engine, _ = _engine_with_targets()
        merges = engine.scan_for_matches()
        assert merges == []

    def test_scan_no_matches_different_sources(self):
        now = time.monotonic()
        t1 = _make_target(
            "ble_aa", source="ble", position=(0.0, 0.0),
        )
        t1.last_seen = now
        t1.first_seen = now - 30.0
        t2 = _make_target(
            "ble_bb", source="ble", position=(500.0, 500.0),
        )
        t2.last_seen = now
        t2.first_seen = now - 30.0
        engine, _ = _engine_with_targets(t1, t2)
        engine.match_threshold = 0.99  # very high threshold
        merges = engine.scan_for_matches()
        assert len(merges) == 0


# ---------------------------------------------------------------------------
# ReIDEngine — statistics
# ---------------------------------------------------------------------------

class TestEngineStats:
    def test_stats(self):
        engine, _ = _engine_with_targets()
        s = engine.stats
        assert "active_profiles" in s
        assert "departed_profiles" in s
        assert "total_merges" in s
        assert s["match_threshold"] == DEFAULT_MATCH_THRESHOLD

    def test_get_profile_cached(self):
        t = _make_target("ble_cached", source="ble")
        engine, _ = _engine_with_targets(t)
        engine.create_profile("ble_cached")
        profile = engine.get_profile("ble_cached")
        assert profile is not None
        assert profile.target_id == "ble_cached"

    def test_get_profile_departed(self):
        t = _make_target("ble_gone", source="ble")
        engine, tracker = _engine_with_targets(t)
        engine.record_departure(t)
        tracker.remove("ble_gone")
        profile = engine.get_profile("ble_gone")
        assert profile is not None

    def test_get_profile_missing(self):
        engine, _ = _engine_with_targets()
        assert engine.get_profile("nonexistent") is None


# ---------------------------------------------------------------------------
# ReIDEngine — weighted scoring
# ---------------------------------------------------------------------------

class TestWeightedScoring:
    def test_default_weights_sum(self):
        total = sum(STRATEGY_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01, f"Weights should sum to ~1.0, got {total}"

    def test_custom_weights(self):
        engine, _ = _engine_with_targets()
        engine.weights = {
            "ble_mac_rotation": 1.0,
            "temporal_cooccurrence": 0.0,
            "spatial_consistency": 0.0,
            "signal_fingerprint": 0.0,
        }
        scores = {
            "ble_mac_rotation": 0.8,
            "temporal_cooccurrence": 0.0,
            "spatial_consistency": 0.0,
            "signal_fingerprint": 0.0,
        }
        result = engine._weighted_score(scores)
        assert abs(result - 0.8) < 0.01

    def test_zero_weights_returns_zero(self):
        engine, _ = _engine_with_targets()
        engine.weights = {}
        scores = {"ble_mac_rotation": 0.9}
        assert engine._weighted_score(scores) == 0.0


# ---------------------------------------------------------------------------
# Integration: imports from tracking __init__
# ---------------------------------------------------------------------------

class TestImports:
    def test_import_from_tracking(self):
        from tritium_lib.tracking import (
            ReIDEngine,
            PersonProfile,
            MatchResult,
            MergeRecord,
            REID_STRATEGY_WEIGHTS,
        )
        assert ReIDEngine is not None
        assert PersonProfile is not None
        assert MatchResult is not None
        assert MergeRecord is not None
        assert isinstance(REID_STRATEGY_WEIGHTS, dict)


# ---------------------------------------------------------------------------
# MergeRecord
# ---------------------------------------------------------------------------

class TestMergeRecord:
    def test_default_id(self):
        r = MergeRecord(primary_id="a", secondary_id="b")
        assert len(r.merge_id) == 12
        assert r.score == 0.0
        assert r.dossier_uuid == ""

    def test_fields(self):
        r = MergeRecord(
            primary_id="ble_aa",
            secondary_id="ble_bb",
            score=0.9,
            dossier_uuid="abc123",
        )
        assert r.primary_id == "ble_aa"
        assert r.secondary_id == "ble_bb"
        assert r.score == 0.9
