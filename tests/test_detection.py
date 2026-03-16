# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine.detection — sensor detection, stealth, countermeasures."""

from __future__ import annotations

import math
import time

import pytest

from tritium_lib.sim_engine.detection import (
    Detection,
    DetectionEngine,
    Sensor,
    SensorType,
    SignatureProfile,
    SIGNATURE_PRESETS,
    _Countermeasure,
)
from tritium_lib.sim_engine.ai.steering import Vec2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sensor(
    sid: str = "s1",
    stype: SensorType = SensorType.VISUAL,
    pos: Vec2 = (0.0, 0.0),
    heading: float = 0.0,
    fov: float = 360.0,
    rng: float = 200.0,
    sens: float = 1.0,
    owner: str = "",
) -> Sensor:
    return Sensor(
        sensor_id=sid, sensor_type=stype, position=pos,
        heading=heading, fov_deg=fov, range_m=rng,
        sensitivity=sens, owner_id=owner,
    )


def _make_engine_with_sensor(**kw) -> tuple[DetectionEngine, Sensor]:
    eng = DetectionEngine()
    s = _make_sensor(**kw)
    eng.sensors.append(s)
    return eng, s


# ---------------------------------------------------------------------------
# SensorType enum
# ---------------------------------------------------------------------------

class TestSensorType:
    def test_all_types_exist(self):
        assert len(SensorType) == 7

    def test_values(self):
        assert SensorType.VISUAL.value == "visual"
        assert SensorType.THERMAL.value == "thermal"
        assert SensorType.ACOUSTIC.value == "acoustic"
        assert SensorType.RADAR.value == "radar"
        assert SensorType.SONAR.value == "sonar"
        assert SensorType.SEISMIC.value == "seismic"
        assert SensorType.RF_PASSIVE.value == "rf_passive"


# ---------------------------------------------------------------------------
# SignatureProfile
# ---------------------------------------------------------------------------

class TestSignatureProfile:
    def test_defaults(self):
        sp = SignatureProfile()
        assert sp.visual == 1.0
        assert sp.thermal == 1.0
        assert sp.acoustic == 1.0
        assert sp.radar == 1.0
        assert sp.rf_emission == 0.0

    def test_get_visual(self):
        sp = SignatureProfile(visual=0.5)
        assert sp.get(SensorType.VISUAL) == 0.5

    def test_get_thermal(self):
        sp = SignatureProfile(thermal=0.3)
        assert sp.get(SensorType.THERMAL) == 0.3

    def test_get_acoustic(self):
        sp = SignatureProfile(acoustic=0.7)
        assert sp.get(SensorType.ACOUSTIC) == 0.7

    def test_get_radar(self):
        sp = SignatureProfile(radar=0.2)
        assert sp.get(SensorType.RADAR) == 0.2

    def test_get_rf_passive(self):
        sp = SignatureProfile(rf_emission=0.9)
        assert sp.get(SensorType.RF_PASSIVE) == 0.9

    def test_get_sonar_uses_acoustic(self):
        sp = SignatureProfile(acoustic=0.4)
        assert sp.get(SensorType.SONAR) == 0.4

    def test_get_seismic_uses_acoustic(self):
        sp = SignatureProfile(acoustic=0.6)
        assert sp.get(SensorType.SEISMIC) == 0.6


# ---------------------------------------------------------------------------
# Signature Presets
# ---------------------------------------------------------------------------

class TestSignaturePresets:
    def test_all_presets_exist(self):
        expected = {
            "infantry", "sniper_ghillie", "vehicle", "tank",
            "helicopter", "drone_small", "submarine_surfaced",
            "submarine_submerged",
        }
        assert set(SIGNATURE_PRESETS.keys()) == expected

    def test_tank_is_loudest(self):
        tank = SIGNATURE_PRESETS["tank"]
        assert tank.visual == 1.0
        assert tank.thermal == 1.0
        assert tank.acoustic == 1.0
        assert tank.radar == 1.0

    def test_sniper_is_stealthy(self):
        sniper = SIGNATURE_PRESETS["sniper_ghillie"]
        assert sniper.visual <= 0.15
        assert sniper.acoustic <= 0.1

    def test_submarine_submerged_invisible(self):
        sub = SIGNATURE_PRESETS["submarine_submerged"]
        assert sub.visual == 0.0
        assert sub.radar == 0.0

    def test_all_presets_are_signature_profiles(self):
        for name, sp in SIGNATURE_PRESETS.items():
            assert isinstance(sp, SignatureProfile), f"{name} is not SignatureProfile"

    def test_drone_small_low_signatures(self):
        d = SIGNATURE_PRESETS["drone_small"]
        assert d.visual < 0.3
        assert d.thermal < 0.2
        assert d.radar < 0.2


# ---------------------------------------------------------------------------
# Sensor dataclass
# ---------------------------------------------------------------------------

class TestSensor:
    def test_defaults(self):
        s = _make_sensor()
        assert s.is_active is True
        assert s.owner_id == ""

    def test_custom_fields(self):
        s = _make_sensor(sid="cam1", stype=SensorType.THERMAL, pos=(10.0, 20.0),
                         heading=1.5, fov=90.0, rng=500.0, sens=0.8, owner="blue_1")
        assert s.sensor_id == "cam1"
        assert s.sensor_type == SensorType.THERMAL
        assert s.position == (10.0, 20.0)
        assert s.fov_deg == 90.0
        assert s.range_m == 500.0
        assert s.sensitivity == 0.8
        assert s.owner_id == "blue_1"


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

class TestDetection:
    def test_creation(self):
        d = Detection(
            detector_id="s1", target_id="e1",
            sensor_type=SensorType.VISUAL,
            confidence=0.7, position_accuracy=5.0,
            timestamp=100.0,
        )
        assert d.is_confirmed is False
        assert d.confidence == 0.7


# ---------------------------------------------------------------------------
# DetectionEngine — basic
# ---------------------------------------------------------------------------

class TestDetectionEngineBasic:
    def test_empty_engine(self):
        eng = DetectionEngine()
        assert eng.sensors == []
        assert eng.signatures == {}
        assert eng.detections == []

    def test_set_signature(self):
        eng = DetectionEngine()
        sig = SignatureProfile(visual=0.5)
        eng.set_signature("e1", sig)
        assert "e1" in eng.signatures

    def test_add_noise(self):
        eng = DetectionEngine()
        eng.add_noise((50.0, 50.0), 0.9, 3.0)
        assert len(eng.noise_sources) == 1
        assert eng.noise_sources[0]["intensity"] == 0.9

    def test_add_noise_clamps(self):
        eng = DetectionEngine()
        eng.add_noise((0.0, 0.0), 2.0, 1.0)
        assert eng.noise_sources[0]["intensity"] == 1.0
        eng.add_noise((0.0, 0.0), -1.0, 1.0)
        assert eng.noise_sources[1]["intensity"] == 0.0


# ---------------------------------------------------------------------------
# DetectionEngine — check_detection
# ---------------------------------------------------------------------------

class TestCheckDetection:
    def test_target_in_range_detected(self):
        eng, sensor = _make_engine_with_sensor(rng=200.0)
        sig = SignatureProfile(visual=1.0)
        det = eng.check_detection(sensor, (50.0, 0.0), sig, {}, target_id="e1")
        assert det is not None
        assert det.confidence > 0.0

    def test_target_out_of_range_not_detected(self):
        eng, sensor = _make_engine_with_sensor(rng=100.0)
        sig = SignatureProfile(visual=1.0)
        det = eng.check_detection(sensor, (200.0, 0.0), sig, {}, target_id="e1")
        assert det is None

    def test_inactive_sensor_no_detection(self):
        eng, sensor = _make_engine_with_sensor()
        sensor.is_active = False
        sig = SignatureProfile(visual=1.0)
        det = eng.check_detection(sensor, (10.0, 0.0), sig, {}, target_id="e1")
        assert det is None

    def test_zero_signature_no_detection(self):
        eng, sensor = _make_engine_with_sensor()
        sig = SignatureProfile(visual=0.0)
        det = eng.check_detection(sensor, (10.0, 0.0), sig, {}, target_id="e1")
        assert det is None

    def test_fov_inside(self):
        # Heading east (0 rad), FOV 90 degrees, target straight ahead
        eng, sensor = _make_engine_with_sensor(heading=0.0, fov=90.0)
        sig = SignatureProfile(visual=1.0)
        det = eng.check_detection(sensor, (50.0, 0.0), sig, {}, target_id="e1")
        assert det is not None

    def test_fov_outside(self):
        # Heading east (0 rad), FOV 90 degrees, target behind
        eng, sensor = _make_engine_with_sensor(heading=0.0, fov=90.0)
        sig = SignatureProfile(visual=1.0)
        det = eng.check_detection(sensor, (-50.0, 0.0), sig, {}, target_id="e1")
        assert det is None

    def test_omnidirectional_detects_behind(self):
        eng, sensor = _make_engine_with_sensor(heading=0.0, fov=360.0)
        sig = SignatureProfile(visual=1.0)
        det = eng.check_detection(sensor, (-50.0, 0.0), sig, {}, target_id="e1")
        assert det is not None

    def test_confidence_decreases_with_distance(self):
        eng, sensor = _make_engine_with_sensor(rng=200.0, sens=1.0)
        sig = SignatureProfile(visual=1.0)
        det_close = eng.check_detection(sensor, (10.0, 0.0), sig, {}, target_id="e1")
        det_far = eng.check_detection(sensor, (150.0, 0.0), sig, {}, target_id="e1")
        assert det_close is not None and det_far is not None
        assert det_close.confidence > det_far.confidence

    def test_high_sensitivity_helps(self):
        eng = DetectionEngine()
        s_low = _make_sensor(sid="low", sens=0.3, rng=200.0)
        s_high = _make_sensor(sid="high", sens=1.0, rng=200.0)
        sig = SignatureProfile(visual=0.3)
        det_low = eng.check_detection(s_low, (50.0, 0.0), sig, {}, target_id="e1")
        det_high = eng.check_detection(s_high, (50.0, 0.0), sig, {}, target_id="e1")
        # High sensitivity should produce higher confidence (or detect when low doesn't)
        if det_low is not None and det_high is not None:
            assert det_high.confidence > det_low.confidence
        elif det_low is None:
            assert det_high is not None

    def test_confirmed_when_high_confidence(self):
        eng, sensor = _make_engine_with_sensor(rng=200.0, sens=1.0)
        sig = SignatureProfile(visual=1.0)
        det = eng.check_detection(sensor, (5.0, 0.0), sig, {}, target_id="e1")
        assert det is not None
        assert det.is_confirmed is True  # very close, full signature

    def test_position_accuracy_worse_at_distance(self):
        eng, sensor = _make_engine_with_sensor(rng=500.0, sens=1.0)
        sig = SignatureProfile(visual=1.0)
        det_close = eng.check_detection(sensor, (10.0, 0.0), sig, {}, target_id="e1")
        det_far = eng.check_detection(sensor, (400.0, 0.0), sig, {}, target_id="e1")
        assert det_close is not None and det_far is not None
        assert det_far.position_accuracy > det_close.position_accuracy


# ---------------------------------------------------------------------------
# Environment modifiers
# ---------------------------------------------------------------------------

class TestEnvironmentModifiers:
    def test_night_reduces_visual(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.VISUAL, rng=200.0)
        sig = SignatureProfile(visual=0.5)
        det_day = eng.check_detection(sensor, (50.0, 0.0), sig, {"is_night": False}, "e1")
        det_night = eng.check_detection(sensor, (50.0, 0.0), sig, {"is_night": True}, "e1")
        if det_day and det_night:
            assert det_night.confidence < det_day.confidence
        elif det_night is None:
            assert det_day is not None  # day detection should still work

    def test_night_improves_thermal(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.THERMAL, rng=200.0)
        sig = SignatureProfile(thermal=0.5)
        det_day = eng.check_detection(sensor, (50.0, 0.0), sig, {"is_night": False}, "e1")
        det_night = eng.check_detection(sensor, (50.0, 0.0), sig, {"is_night": True}, "e1")
        assert det_day is not None and det_night is not None
        assert det_night.confidence > det_day.confidence

    def test_fog_reduces_visual(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.VISUAL, rng=200.0)
        sig = SignatureProfile(visual=0.8)
        det_clear = eng.check_detection(sensor, (50.0, 0.0), sig, {"weather": "clear"}, "e1")
        det_fog = eng.check_detection(sensor, (50.0, 0.0), sig, {"weather": "fog"}, "e1")
        if det_clear and det_fog:
            assert det_fog.confidence < det_clear.confidence

    def test_rain_reduces_visual(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.VISUAL, rng=200.0)
        sig = SignatureProfile(visual=0.8)
        det_clear = eng.check_detection(sensor, (50.0, 0.0), sig, {"weather": "clear"}, "e1")
        det_rain = eng.check_detection(sensor, (50.0, 0.0), sig, {"weather": "rain"}, "e1")
        assert det_clear is not None and det_rain is not None
        assert det_rain.confidence < det_clear.confidence

    def test_cover_reduces_detection(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.VISUAL, rng=200.0)
        sig = SignatureProfile(visual=0.8)
        det_open = eng.check_detection(sensor, (50.0, 0.0), sig, {"cover": 0.0}, "e1")
        det_cover = eng.check_detection(sensor, (50.0, 0.0), sig, {"cover": 0.8}, "e1")
        assert det_open is not None
        if det_cover is not None:
            assert det_cover.confidence < det_open.confidence

    def test_wind_reduces_acoustic(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.ACOUSTIC, rng=200.0)
        sig = SignatureProfile(acoustic=0.5)
        det_calm = eng.check_detection(sensor, (50.0, 0.0), sig, {"wind_speed": 0.0}, "e1")
        det_windy = eng.check_detection(sensor, (50.0, 0.0), sig, {"wind_speed": 15.0}, "e1")
        if det_calm and det_windy:
            assert det_windy.confidence < det_calm.confidence

    def test_storm_reduces_radar(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.RADAR, rng=200.0)
        sig = SignatureProfile(radar=0.8)
        det_clear = eng.check_detection(sensor, (50.0, 0.0), sig, {"weather": "clear"}, "e1")
        det_storm = eng.check_detection(sensor, (50.0, 0.0), sig, {"weather": "storm"}, "e1")
        assert det_clear is not None and det_storm is not None
        assert det_storm.confidence < det_clear.confidence

    def test_rf_passive_unaffected_by_weather(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.RF_PASSIVE, rng=200.0)
        sig = SignatureProfile(rf_emission=0.8)
        det_clear = eng.check_detection(sensor, (50.0, 0.0), sig, {"weather": "clear"}, "e1")
        det_storm = eng.check_detection(sensor, (50.0, 0.0), sig, {"weather": "storm"}, "e1")
        assert det_clear is not None and det_storm is not None
        assert abs(det_clear.confidence - det_storm.confidence) < 0.01


# ---------------------------------------------------------------------------
# Countermeasures
# ---------------------------------------------------------------------------

class TestCountermeasures:
    def test_smoke_reduces_visual(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.VISUAL, rng=200.0)
        sig = SignatureProfile(visual=1.0)
        target_pos = (50.0, 0.0)
        det_before = eng.check_detection(sensor, target_pos, sig, {}, "e1")
        eng.deploy_smoke(target_pos, radius=20.0)
        det_after = eng.check_detection(sensor, target_pos, sig, {}, "e1")
        assert det_before is not None
        if det_after is not None:
            assert det_after.confidence < det_before.confidence

    def test_smoke_no_effect_on_thermal(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.THERMAL, rng=200.0)
        sig = SignatureProfile(thermal=1.0)
        target_pos = (50.0, 0.0)
        det_before = eng.check_detection(sensor, target_pos, sig, {}, "e1")
        eng.deploy_smoke(target_pos, radius=20.0)
        det_after = eng.check_detection(sensor, target_pos, sig, {}, "e1")
        assert det_before is not None and det_after is not None
        assert abs(det_before.confidence - det_after.confidence) < 0.01

    def test_chaff_reduces_radar(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.RADAR, rng=200.0)
        sig = SignatureProfile(radar=1.0)
        target_pos = (50.0, 0.0)
        det_before = eng.check_detection(sensor, target_pos, sig, {}, "e1")
        eng.deploy_chaff(target_pos, radius=20.0)
        det_after = eng.check_detection(sensor, target_pos, sig, {}, "e1")
        assert det_before is not None
        if det_after is not None:
            assert det_after.confidence < det_before.confidence

    def test_chaff_no_effect_on_visual(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.VISUAL, rng=200.0)
        sig = SignatureProfile(visual=1.0)
        target_pos = (50.0, 0.0)
        det_before = eng.check_detection(sensor, target_pos, sig, {}, "e1")
        eng.deploy_chaff(target_pos, radius=20.0)
        det_after = eng.check_detection(sensor, target_pos, sig, {}, "e1")
        assert det_before is not None and det_after is not None
        assert abs(det_before.confidence - det_after.confidence) < 0.01

    def test_flare_reduces_thermal(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.THERMAL, rng=200.0)
        sig = SignatureProfile(thermal=1.0)
        target_pos = (50.0, 0.0)
        det_before = eng.check_detection(sensor, target_pos, sig, {}, "e1")
        eng.deploy_flare(target_pos)
        det_after = eng.check_detection(sensor, target_pos, sig, {}, "e1")
        assert det_before is not None
        if det_after is not None:
            assert det_after.confidence < det_before.confidence

    def test_countermeasure_outside_radius_no_effect(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.VISUAL, rng=300.0)
        sig = SignatureProfile(visual=1.0)
        target_pos = (200.0, 0.0)
        det_before = eng.check_detection(sensor, target_pos, sig, {}, "e1")
        eng.deploy_smoke((0.0, 0.0), radius=10.0)  # smoke far from target
        det_after = eng.check_detection(sensor, target_pos, sig, {}, "e1")
        assert det_before is not None and det_after is not None
        assert abs(det_before.confidence - det_after.confidence) < 0.01

    def test_radio_silent(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.RF_PASSIVE, rng=200.0)
        sig = SignatureProfile(rf_emission=1.0)
        det_before = eng.check_detection(sensor, (50.0, 0.0), sig, {}, "e1")
        assert det_before is not None
        eng.go_radio_silent("e1")
        det_after = eng.check_detection(sensor, (50.0, 0.0), sig, {}, "e1")
        assert det_after is None

    def test_break_radio_silence(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.RF_PASSIVE, rng=200.0)
        sig = SignatureProfile(rf_emission=1.0)
        eng.go_radio_silent("e1")
        eng.break_radio_silence("e1")
        det = eng.check_detection(sensor, (50.0, 0.0), sig, {}, "e1")
        assert det is not None


# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------

class TestTick:
    def test_tick_detects_entity(self):
        eng = DetectionEngine()
        s = _make_sensor(sid="s1", owner="blue_1")
        eng.sensors.append(s)
        eng.set_signature("e1", SignatureProfile(visual=1.0))
        results = eng.tick(0.1, {"e1": (50.0, 0.0)}, {})
        assert len(results) == 1
        assert results[0].target_id == "e1"

    def test_tick_skips_own_entity(self):
        eng = DetectionEngine()
        s = _make_sensor(sid="s1", owner="blue_1")
        eng.sensors.append(s)
        eng.set_signature("blue_1", SignatureProfile(visual=1.0))
        results = eng.tick(0.1, {"blue_1": (50.0, 0.0)}, {})
        assert len(results) == 0

    def test_tick_skips_unknown_signature(self):
        eng = DetectionEngine()
        s = _make_sensor(sid="s1", owner="blue_1")
        eng.sensors.append(s)
        # No signature set for "e1"
        results = eng.tick(0.1, {"e1": (50.0, 0.0)}, {})
        assert len(results) == 0

    def test_tick_multi_sensor_confirms(self):
        eng = DetectionEngine()
        s_vis = _make_sensor(sid="vis", stype=SensorType.VISUAL, owner="blue_1")
        s_therm = _make_sensor(sid="therm", stype=SensorType.THERMAL, owner="blue_1")
        eng.sensors.extend([s_vis, s_therm])
        eng.set_signature("e1", SignatureProfile(visual=1.0, thermal=1.0))
        results = eng.tick(0.1, {"e1": (50.0, 0.0)}, {})
        # Merged — only one detection for e1
        assert len(results) == 1
        assert results[0].is_confirmed is True

    def test_tick_decays_countermeasures(self):
        eng = DetectionEngine()
        eng.deploy_smoke((50.0, 0.0), radius=20.0, duration=0.5)
        assert len(eng._countermeasures) == 1
        eng.tick(1.0, {}, {})  # dt > duration
        assert len(eng._countermeasures) == 0

    def test_tick_decays_noise(self):
        eng = DetectionEngine()
        eng.add_noise((50.0, 0.0), 0.8, 0.5)
        assert len(eng.noise_sources) == 1
        eng.tick(1.0, {}, {})
        assert len(eng.noise_sources) == 0

    def test_tick_multiple_entities(self):
        eng = DetectionEngine()
        s = _make_sensor(sid="s1", owner="blue_1", rng=500.0)
        eng.sensors.append(s)
        eng.set_signature("e1", SignatureProfile(visual=1.0))
        eng.set_signature("e2", SignatureProfile(visual=0.8))
        results = eng.tick(0.1, {"e1": (50.0, 0.0), "e2": (100.0, 0.0)}, {})
        target_ids = {r.target_id for r in results}
        assert "e1" in target_ids
        assert "e2" in target_ids

    def test_tick_inactive_sensor_skipped(self):
        eng = DetectionEngine()
        s = _make_sensor(sid="s1", owner="blue_1")
        s.is_active = False
        eng.sensors.append(s)
        eng.set_signature("e1", SignatureProfile(visual=1.0))
        results = eng.tick(0.1, {"e1": (50.0, 0.0)}, {})
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Detection map
# ---------------------------------------------------------------------------

class TestDetectionMap:
    def test_get_detection_map_filters_by_alliance(self):
        eng = DetectionEngine()
        s_blue = _make_sensor(sid="s_blue", owner="blue_1")
        s_red = _make_sensor(sid="s_red", owner="red_1")
        eng.sensors.extend([s_blue, s_red])
        eng.set_signature("e1", SignatureProfile(visual=1.0))
        eng.tick(0.1, {"e1": (50.0, 0.0)}, {})
        blue_map = eng.get_detection_map("blue")
        red_map = eng.get_detection_map("red")
        # Both sides should see e1
        assert "e1" in blue_map
        assert "e1" in red_map

    def test_detection_map_empty_for_unknown_alliance(self):
        eng = DetectionEngine()
        s = _make_sensor(sid="s1", owner="blue_1")
        eng.sensors.append(s)
        eng.set_signature("e1", SignatureProfile(visual=1.0))
        eng.tick(0.1, {"e1": (50.0, 0.0)}, {})
        green_map = eng.get_detection_map("green")
        assert len(green_map) == 0

    def test_detection_map_structure(self):
        eng = DetectionEngine()
        s = _make_sensor(sid="s1", owner="blue_1")
        eng.sensors.append(s)
        eng.set_signature("e1", SignatureProfile(visual=1.0))
        eng.tick(0.1, {"e1": (50.0, 0.0)}, {})
        dmap = eng.get_detection_map("blue")
        assert "e1" in dmap
        entry = dmap["e1"]
        assert "target_id" in entry
        assert "sensor_type" in entry
        assert "confidence" in entry
        assert "position_accuracy" in entry
        assert "is_confirmed" in entry


# ---------------------------------------------------------------------------
# Three.js export
# ---------------------------------------------------------------------------

class TestThreeJsExport:
    def test_empty_engine_export(self):
        eng = DetectionEngine()
        out = eng.to_three_js()
        assert out == {"sensors": [], "detections": [], "noise_sources": []}

    def test_sensor_export(self):
        eng = DetectionEngine()
        eng.sensors.append(_make_sensor(sid="cam1", stype=SensorType.VISUAL,
                                        pos=(10.0, 20.0), fov=90.0, rng=150.0))
        out = eng.to_three_js()
        assert len(out["sensors"]) == 1
        s = out["sensors"][0]
        assert s["id"] == "cam1"
        assert s["x"] == 10.0
        assert s["y"] == 20.0
        assert s["fov"] == 90.0
        assert s["range"] == 150.0
        assert s["type"] == "visual"
        assert s["cone_color"] == "#00f0ff33"
        assert s["active"] is True

    def test_detection_export(self):
        eng = DetectionEngine()
        s = _make_sensor(sid="s1", owner="blue_1")
        eng.sensors.append(s)
        eng.set_signature("e1", SignatureProfile(visual=1.0))
        eng.tick(0.1, {"e1": (50.0, 0.0)}, {})
        out = eng.to_three_js()
        assert len(out["detections"]) >= 1
        d = out["detections"][0]
        assert d["target_id"] == "e1"
        assert "confidence" in d
        assert "accuracy_circle" in d
        assert "sensor_type" in d
        assert "confirmed" in d

    def test_noise_export(self):
        eng = DetectionEngine()
        eng.add_noise((100.0, 60.0), 0.8, 5.0)
        out = eng.to_three_js()
        assert len(out["noise_sources"]) == 1
        n = out["noise_sources"][0]
        assert n["x"] == 100.0
        assert n["y"] == 60.0
        assert n["intensity"] == 0.8

    def test_all_sensor_types_have_colors(self):
        eng = DetectionEngine()
        for st in SensorType:
            eng.sensors.append(_make_sensor(sid=st.value, stype=st, pos=(0.0, 0.0)))
        out = eng.to_three_js()
        for s in out["sensors"]:
            assert s["cone_color"].endswith("33"), f"{s['type']} missing color"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_range_sensor(self):
        eng, sensor = _make_engine_with_sensor(rng=0.0)
        sig = SignatureProfile(visual=1.0)
        det = eng.check_detection(sensor, (0.0, 0.0), sig, {}, "e1")
        # Zero range sensor cannot detect anything
        assert det is None

    def test_sensor_at_same_position_as_target(self):
        eng, sensor = _make_engine_with_sensor(rng=100.0)
        sig = SignatureProfile(visual=1.0)
        det = eng.check_detection(sensor, (0.0, 0.0), sig, {}, "e1")
        assert det is not None
        assert det.confidence > 0.9  # point-blank

    def test_very_low_signature_not_detected(self):
        eng, sensor = _make_engine_with_sensor(rng=200.0, sens=0.5)
        sig = SignatureProfile(visual=0.01)
        det = eng.check_detection(sensor, (150.0, 0.0), sig, {}, "e1")
        # Very low signature at medium range with medium sensitivity
        # Should likely be below threshold
        assert det is None

    def test_multiple_countermeasures_stack(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.VISUAL, rng=200.0)
        sig = SignatureProfile(visual=1.0)
        target_pos = (50.0, 0.0)
        det_before = eng.check_detection(sensor, target_pos, sig, {}, "e1")
        eng.deploy_smoke(target_pos, radius=20.0)
        eng.deploy_smoke(target_pos, radius=20.0)
        det_after = eng.check_detection(sensor, target_pos, sig, {}, "e1")
        assert det_before is not None
        if det_after is not None:
            assert det_after.confidence < det_before.confidence

    def test_fov_boundary(self):
        # Target exactly at FOV edge
        eng, sensor = _make_engine_with_sensor(heading=0.0, fov=90.0, rng=200.0)
        sig = SignatureProfile(visual=1.0)
        # 44 degrees off center — just inside 90/2=45
        angle_rad = math.radians(44.0)
        target = (50.0 * math.cos(angle_rad), 50.0 * math.sin(angle_rad))
        det = eng.check_detection(sensor, target, sig, {}, "e1")
        assert det is not None

    def test_fov_just_outside(self):
        eng, sensor = _make_engine_with_sensor(heading=0.0, fov=90.0, rng=200.0)
        sig = SignatureProfile(visual=1.0)
        # 46 degrees off center — just outside
        angle_rad = math.radians(46.0)
        target = (50.0 * math.cos(angle_rad), 50.0 * math.sin(angle_rad))
        det = eng.check_detection(sensor, target, sig, {}, "e1")
        assert det is None

    def test_thermal_sensor_with_visual_signature_only(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.THERMAL, rng=200.0)
        sig = SignatureProfile(visual=1.0, thermal=0.0)
        det = eng.check_detection(sensor, (50.0, 0.0), sig, {}, "e1")
        assert det is None

    def test_rf_passive_detects_emission(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.RF_PASSIVE, rng=200.0)
        sig = SignatureProfile(rf_emission=0.8)
        det = eng.check_detection(sensor, (50.0, 0.0), sig, {}, "e1")
        assert det is not None

    def test_rf_passive_no_emission(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.RF_PASSIVE, rng=200.0)
        sig = SignatureProfile(rf_emission=0.0)
        det = eng.check_detection(sensor, (50.0, 0.0), sig, {}, "e1")
        assert det is None

    def test_combined_weather_and_night(self):
        eng, sensor = _make_engine_with_sensor(stype=SensorType.VISUAL, rng=200.0)
        sig = SignatureProfile(visual=1.0)
        env = {"is_night": True, "weather": "fog"}
        det = eng.check_detection(sensor, (50.0, 0.0), sig, env, "e1")
        # Night + fog = severe visual penalty — may not detect at all
        if det is not None:
            assert det.confidence < 0.3
