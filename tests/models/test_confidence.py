# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for confidence decay models."""

import math

from tritium_lib.models.confidence import (
    ConfidenceModel,
    DEFAULT_HALF_LIVES,
    SourceType,
)


def test_source_type_values():
    assert SourceType.BLE.value == "ble"
    assert SourceType.MESH.value == "mesh"
    assert SourceType.YOLO.value == "yolo"
    assert SourceType.SIMULATION.value == "simulation"


def test_default_half_lives():
    assert DEFAULT_HALF_LIVES["ble"] == 30.0
    assert DEFAULT_HALF_LIVES["mesh"] == 120.0
    assert DEFAULT_HALF_LIVES["yolo"] == 15.0
    assert DEFAULT_HALF_LIVES["simulation"] == 0.0


def test_no_elapsed_returns_initial():
    model = ConfidenceModel()
    assert model.decay("ble", 0.85, 0.0) == 0.85


def test_ble_decays_at_half_life():
    model = ConfidenceModel()
    # After one half-life (30s), confidence should be ~50% of initial
    result = model.decay("ble", 1.0, 30.0)
    assert abs(result - 0.5) < 0.01


def test_mesh_decays_slower():
    model = ConfidenceModel()
    # After 30s, mesh should retain much more confidence than BLE
    ble_conf = model.decay("ble", 1.0, 30.0)
    mesh_conf = model.decay("mesh", 1.0, 30.0)
    assert mesh_conf > ble_conf
    assert mesh_conf > 0.8  # mesh barely decayed at 30s


def test_simulation_never_decays():
    model = ConfidenceModel()
    result = model.decay("simulation", 1.0, 9999.0)
    assert result == 1.0


def test_stale_detection():
    model = ConfidenceModel()
    # BLE target after 5 minutes should be stale
    assert model.is_stale("ble", 300.0)
    # BLE target after 5 seconds should not be stale
    assert not model.is_stale("ble", 5.0)


def test_time_to_stale():
    model = ConfidenceModel()
    t = model.time_to_stale("ble", initial=1.0)
    # At exactly t, confidence equals min_confidence (boundary)
    # Slightly past t should be stale
    decayed_past = model.decay("ble", 1.0, t + 1.0)
    assert decayed_past == 0.0
    # Just before should still be alive
    decayed_before = model.decay("ble", 1.0, t - 1.0)
    assert decayed_before > 0.0


def test_time_to_stale_simulation():
    model = ConfidenceModel()
    assert model.time_to_stale("simulation") == float("inf")


def test_custom_half_lives():
    model = ConfidenceModel(half_lives={"ble": 10.0})
    # BLE should decay faster with 10s half-life
    result = model.decay("ble", 1.0, 10.0)
    assert abs(result - 0.5) < 0.01


def test_set_half_life():
    model = ConfidenceModel()
    model.set_half_life("ble", 60.0)
    assert model.get_half_life("ble") == 60.0


def test_roundtrip_serialization():
    model = ConfidenceModel(min_confidence=0.1)
    d = model.to_dict()
    restored = ConfidenceModel.from_dict(d)
    assert restored.min_confidence == 0.1
    assert restored.half_lives["ble"] == model.half_lives["ble"]


def test_unknown_source_uses_manual_fallback():
    model = ConfidenceModel()
    result = model.decay("unknown_sensor", 1.0, 300.0)
    # Should use manual half-life (300s), so at 300s = 50%
    assert abs(result - 0.5) < 0.01


def test_exponential_decay_formula():
    model = ConfidenceModel()
    half_life = 30.0
    initial = 0.8
    elapsed = 45.0
    expected = initial * math.exp(-math.log(2) / half_life * elapsed)
    result = model.decay("ble", initial, elapsed)
    assert abs(result - expected) < 0.001


def test_import_from_top_level():
    from tritium_lib.models import ConfidenceModel as CM, SourceType as ST
    assert CM is ConfidenceModel
    assert ST is SourceType
