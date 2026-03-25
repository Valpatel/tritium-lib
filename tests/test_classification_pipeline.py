# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.classification — multi-sensor target classification pipeline."""

import pytest

from tritium_lib.classification import (
    BLETypeClassifier,
    BehaviorClassifier,
    ClassificationPipeline,
    ClassificationResult,
    ClassificationVote,
    EnsembleClassifier,
    SpeedClassifier,
    TARGET_TYPES,
    TargetObservation,
    TargetType,
    TimeClassifier,
    WiFiTypeClassifier,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def pipeline() -> ClassificationPipeline:
    return ClassificationPipeline()


@pytest.fixture
def ble_clf() -> BLETypeClassifier:
    return BLETypeClassifier()


@pytest.fixture
def wifi_clf() -> WiFiTypeClassifier:
    return WiFiTypeClassifier()


@pytest.fixture
def speed_clf() -> SpeedClassifier:
    return SpeedClassifier()


@pytest.fixture
def behavior_clf() -> BehaviorClassifier:
    return BehaviorClassifier()


@pytest.fixture
def time_clf() -> TimeClassifier:
    return TimeClassifier()


# ── TargetType enum ───────────────────────────────────────────────────────

class TestTargetType:
    def test_all_types_in_enum(self):
        assert len(TargetType) == 7
        for t in TARGET_TYPES:
            assert t in TargetType._value2member_map_

    def test_target_types_list_matches_enum(self):
        assert TARGET_TYPES == [t.value for t in TargetType]


# ── ClassificationVote ────────────────────────────────────────────────────

class TestClassificationVote:
    def test_defaults(self):
        v = ClassificationVote()
        assert v.target_type == "unknown"
        assert v.confidence == 0.0
        assert v.source == ""

    def test_to_dict(self):
        v = ClassificationVote(
            target_type="person", subtype="pedestrian",
            confidence=0.8, evidence="walking", source="speed",
        )
        d = v.to_dict()
        assert d["target_type"] == "person"
        assert d["confidence"] == 0.8
        assert d["source"] == "speed"


# ── ClassificationResult ─────────────────────────────────────────────────

class TestClassificationResult:
    def test_defaults(self):
        r = ClassificationResult()
        assert r.target_type == "unknown"
        assert r.confidence == 0.0
        assert r.votes == []

    def test_to_dict_includes_votes(self):
        v = ClassificationVote(target_type="vehicle", confidence=0.7, source="speed")
        r = ClassificationResult(
            target_type="vehicle", confidence=0.7, votes=[v],
            evidence=["fast mover"],
        )
        d = r.to_dict()
        assert d["target_type"] == "vehicle"
        assert len(d["votes"]) == 1
        assert d["evidence"] == ["fast mover"]


# ── BLETypeClassifier ────────────────────────────────────────────────────

class TestBLETypeClassifier:
    def test_iphone_name(self, ble_clf):
        obs = TargetObservation(ble_name="iPhone 15 Pro")
        vote = ble_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "mobile_device"
        assert vote.subtype == "phone"
        assert vote.confidence >= 0.85

    def test_airpods_name(self, ble_clf):
        obs = TargetObservation(ble_name="AirPods Pro")
        vote = ble_clf.classify(obs)
        assert vote is not None
        assert vote.subtype == "earbuds"

    def test_tesla_name(self, ble_clf):
        obs = TargetObservation(ble_name="Tesla Model Y")
        vote = ble_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "vehicle"
        assert vote.subtype == "car"

    def test_esp32_name(self, ble_clf):
        obs = TargetObservation(ble_name="ESP32-Weather")
        vote = ble_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "fixed_device"
        assert vote.subtype == "sensor"

    def test_appearance_code_phone(self, ble_clf):
        obs = TargetObservation(ble_appearance=64)
        vote = ble_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "mobile_device"
        assert vote.subtype == "phone"

    def test_manufacturer_espressif(self, ble_clf):
        obs = TargetObservation(ble_manufacturer="Espressif Systems")
        vote = ble_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "fixed_device"

    def test_no_ble_data_returns_none(self, ble_clf):
        obs = TargetObservation()
        vote = ble_clf.classify(obs)
        assert vote is None

    def test_device_type_hint_fallback(self, ble_clf):
        obs = TargetObservation(ble_mac="AA:BB:CC:DD:EE:FF", device_type_hint="phone")
        vote = ble_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "mobile_device"
        assert vote.subtype == "phone"


# ── WiFiTypeClassifier ───────────────────────────────────────────────────

class TestWiFiTypeClassifier:
    def test_iphone_ssid(self, wifi_clf):
        obs = TargetObservation(wifi_ssid="iPhone")
        vote = wifi_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "mobile_device"
        assert vote.subtype == "phone"

    def test_tesla_ssid(self, wifi_clf):
        obs = TargetObservation(wifi_ssid="TeslaGuest")
        vote = wifi_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "vehicle"

    def test_high_probe_count(self, wifi_clf):
        obs = TargetObservation(wifi_probe_count=10)
        vote = wifi_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "mobile_device"

    def test_low_probe_count(self, wifi_clf):
        obs = TargetObservation(wifi_probe_count=2)
        vote = wifi_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "unknown"

    def test_randomized_mac(self, wifi_clf):
        obs = TargetObservation(wifi_is_randomized=True)
        vote = wifi_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "mobile_device"

    def test_no_wifi_data_returns_none(self, wifi_clf):
        obs = TargetObservation()
        vote = wifi_clf.classify(obs)
        assert vote is None


# ── SpeedClassifier ──────────────────────────────────────────────────────

class TestSpeedClassifier:
    def test_stationary(self, speed_clf):
        obs = TargetObservation(speed_mps=0.1)
        vote = speed_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "unknown"

    def test_walking_speed(self, speed_clf):
        obs = TargetObservation(avg_speed_mps=1.4)
        vote = speed_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "person"
        assert vote.subtype == "pedestrian"

    def test_running_speed(self, speed_clf):
        obs = TargetObservation(avg_speed_mps=3.5)
        vote = speed_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "person"
        assert vote.subtype == "runner"

    def test_bicycle_speed(self, speed_clf):
        obs = TargetObservation(avg_speed_mps=7.0)
        vote = speed_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "bicycle"

    def test_vehicle_speed(self, speed_clf):
        obs = TargetObservation(avg_speed_mps=25.0)
        vote = speed_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "vehicle"
        assert vote.subtype == "car"

    def test_bicycle_speed_with_high_max(self, speed_clf):
        """If current speed is bicycle-range but max exceeds it, call it vehicle."""
        obs = TargetObservation(avg_speed_mps=8.0, max_speed_mps=20.0)
        vote = speed_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "vehicle"

    def test_no_speed_returns_none(self, speed_clf):
        obs = TargetObservation()
        vote = speed_clf.classify(obs)
        assert vote is None

    def test_prefers_avg_over_instantaneous(self, speed_clf):
        """avg_speed_mps should be used preferentially."""
        obs = TargetObservation(speed_mps=25.0, avg_speed_mps=1.3)
        vote = speed_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "person"


# ── BehaviorClassifier ───────────────────────────────────────────────────

class TestBehaviorClassifier:
    def test_loitering(self, behavior_clf):
        obs = TargetObservation(movement_pattern="loitering")
        vote = behavior_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "person"
        assert vote.subtype == "loiterer"

    def test_erratic(self, behavior_clf):
        obs = TargetObservation(movement_pattern="erratic")
        vote = behavior_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "animal"

    def test_long_stationary_fixed_device(self, behavior_clf):
        obs = TargetObservation(
            is_stationary=True,
            dwell_seconds=7200.0,  # 2 hours
        )
        vote = behavior_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "fixed_device"

    def test_patrol_walking(self, behavior_clf):
        obs = TargetObservation(movement_pattern="patrol", avg_speed_mps=1.5)
        vote = behavior_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "person"

    def test_patrol_driving(self, behavior_clf):
        obs = TargetObservation(movement_pattern="patrol", avg_speed_mps=15.0)
        vote = behavior_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "vehicle"

    def test_transit_low_speed(self, behavior_clf):
        obs = TargetObservation(movement_pattern="transit", avg_speed_mps=1.2)
        vote = behavior_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "person"
        assert vote.subtype == "commuter"

    def test_transit_high_speed(self, behavior_clf):
        obs = TargetObservation(movement_pattern="transit", avg_speed_mps=20.0)
        vote = behavior_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "vehicle"

    def test_no_behavior_returns_none(self, behavior_clf):
        obs = TargetObservation()
        vote = behavior_clf.classify(obs)
        assert vote is None


# ── TimeClassifier ────────────────────────────────────────────────────────

class TestTimeClassifier:
    def test_night_frequent_visitor_is_resident(self, time_clf):
        obs = TargetObservation(hour_of_day=23, visit_count=10)
        vote = time_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "person"
        assert vote.subtype == "resident"

    def test_rush_hour_weekday_commuter(self, time_clf):
        obs = TargetObservation(hour_of_day=8, day_of_week=2, visit_count=5)
        vote = time_clf.classify(obs)
        assert vote is not None
        assert vote.subtype == "commuter"

    def test_single_visit_night_inconclusive(self, time_clf):
        obs = TargetObservation(hour_of_day=2, visit_count=1)
        vote = time_clf.classify(obs)
        assert vote is not None
        assert vote.target_type == "unknown"

    def test_no_hour_returns_none(self, time_clf):
        obs = TargetObservation()
        vote = time_clf.classify(obs)
        assert vote is None

    def test_frequent_visitor_daytime(self, time_clf):
        obs = TargetObservation(hour_of_day=14, visit_count=15)
        vote = time_clf.classify(obs)
        assert vote is not None
        assert vote.subtype == "resident"


# ── EnsembleClassifier ───────────────────────────────────────────────────

class TestEnsembleClassifier:
    def test_empty_classifiers(self):
        ens = EnsembleClassifier([])
        result = ens.classify(TargetObservation())
        assert result.target_type == "unknown"
        assert result.confidence == 0.0

    def test_single_classifier(self):
        ens = EnsembleClassifier([SpeedClassifier()])
        obs = TargetObservation(avg_speed_mps=1.4)
        result = ens.classify(obs)
        assert result.target_type == "person"

    def test_multi_source_agreement_boost(self):
        """Two classifiers agreeing should boost confidence."""
        ens = EnsembleClassifier([SpeedClassifier(), BehaviorClassifier()])
        obs = TargetObservation(
            avg_speed_mps=1.4,
            movement_pattern="loitering",
        )
        result = ens.classify(obs)
        assert result.target_type == "person"
        # With two agreeing votes, should have agreement boost
        assert result.confidence > 0.55

    def test_add_remove_classifier(self):
        ens = EnsembleClassifier()
        ens.add_classifier(SpeedClassifier())
        assert len(ens.classifiers) == 1
        assert ens.remove_classifier("speed") is True
        assert len(ens.classifiers) == 0
        assert ens.remove_classifier("nonexistent") is False

    def test_conflicting_votes_highest_conf_wins(self):
        """When classifiers disagree, the one with highest total score wins."""

        class AlwaysVehicle:
            name = "always_vehicle"
            def classify(self, obs):
                return ClassificationVote(
                    target_type="vehicle", confidence=0.9, source=self.name,
                )

        class AlwaysPerson:
            name = "always_person"
            def classify(self, obs):
                return ClassificationVote(
                    target_type="person", confidence=0.4, source=self.name,
                )

        ens = EnsembleClassifier([AlwaysVehicle(), AlwaysPerson()])
        result = ens.classify(TargetObservation())
        assert result.target_type == "vehicle"

    def test_broken_classifier_handled_gracefully(self):
        """A classifier that raises should not break the ensemble."""

        class BrokenClassifier:
            name = "broken"
            def classify(self, obs):
                raise RuntimeError("boom")

        ens = EnsembleClassifier([BrokenClassifier(), SpeedClassifier()])
        obs = TargetObservation(avg_speed_mps=1.4)
        result = ens.classify(obs)
        # SpeedClassifier still works
        assert result.target_type == "person"

    def test_unknown_loses_to_concrete(self):
        """When unknown and a real type tie on score, the real type wins."""

        class LowUnknown:
            name = "low_unknown"
            def classify(self, obs):
                return ClassificationVote(
                    target_type="unknown", confidence=0.50, source=self.name,
                )

        class LowAnimal:
            name = "low_animal"
            def classify(self, obs):
                return ClassificationVote(
                    target_type="animal", confidence=0.50, source=self.name,
                )

        ens = EnsembleClassifier([LowUnknown(), LowAnimal()])
        result = ens.classify(TargetObservation())
        assert result.target_type == "animal"


# ── ClassificationPipeline ───────────────────────────────────────────────

class TestClassificationPipeline:
    def test_default_has_five_classifiers(self, pipeline):
        assert len(pipeline.ensemble.classifiers) == 5

    def test_classify_person_from_phone_ble(self, pipeline):
        """Phone BLE name → mobile_device (carried by a person)."""
        obs = TargetObservation(ble_name="iPhone 15")
        result = pipeline.classify(obs)
        assert result.target_type == "mobile_device"
        assert result.subtype == "phone"
        assert result.confidence > 0

    def test_classify_vehicle_from_speed(self, pipeline):
        obs = TargetObservation(avg_speed_mps=30.0)
        result = pipeline.classify(obs)
        assert result.target_type == "vehicle"

    def test_classify_pedestrian_multi_signal(self, pipeline):
        """Walking speed + loitering + rush-hour weekday → person."""
        obs = TargetObservation(
            avg_speed_mps=1.3,
            movement_pattern="loitering",
            hour_of_day=8,
            day_of_week=1,
            visit_count=5,
        )
        result = pipeline.classify(obs)
        assert result.target_type == "person"
        assert len(result.votes) >= 3  # speed, behavior, time all vote

    def test_classify_many(self, pipeline):
        obs_list = [
            TargetObservation(avg_speed_mps=1.4),
            TargetObservation(avg_speed_mps=25.0),
            TargetObservation(ble_name="AirPods Pro"),
        ]
        results = pipeline.classify_many(obs_list)
        assert len(results) == 3
        assert results[0].target_type == "person"
        assert results[1].target_type == "vehicle"
        assert results[2].target_type == "mobile_device"

    def test_classify_empty_observation(self, pipeline):
        """Empty observation → unknown with no votes or very low confidence."""
        obs = TargetObservation()
        result = pipeline.classify(obs)
        # With no data, classifiers abstain
        assert result.target_type == "unknown"

    def test_add_custom_classifier(self, pipeline):
        class Dummy:
            name = "dummy"
            def classify(self, obs):
                return ClassificationVote(
                    target_type="animal", subtype="dog",
                    confidence=0.99, source=self.name,
                )

        pipeline.add_classifier(Dummy())
        assert len(pipeline.ensemble.classifiers) == 6
        result = pipeline.classify(TargetObservation())
        assert result.target_type == "animal"

    def test_result_evidence_populated(self, pipeline):
        obs = TargetObservation(ble_name="iPhone 15", avg_speed_mps=1.4)
        result = pipeline.classify(obs)
        assert len(result.evidence) >= 2

    def test_result_to_dict(self, pipeline):
        obs = TargetObservation(avg_speed_mps=1.4)
        result = pipeline.classify(obs)
        d = result.to_dict()
        assert "target_type" in d
        assert "confidence" in d
        assert "votes" in d
        assert isinstance(d["votes"], list)

    def test_custom_classifiers_override_defaults(self):
        """Passing classifiers= overrides the default set."""
        p = ClassificationPipeline(classifiers=[SpeedClassifier()])
        assert len(p.ensemble.classifiers) == 1

    def test_bicycle_classification(self, pipeline):
        obs = TargetObservation(avg_speed_mps=8.0)
        result = pipeline.classify(obs)
        assert result.target_type == "bicycle"

    def test_fixed_device_from_long_dwell(self, pipeline):
        obs = TargetObservation(
            is_stationary=True,
            dwell_seconds=7200.0,
            ble_name="ESP32-Sensor",
        )
        result = pipeline.classify(obs)
        assert result.target_type == "fixed_device"
