# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.intelligence.acoustic_classifier."""

from tritium_lib.intelligence.acoustic_classifier import (
    AcousticClassifier,
    AcousticEvent,
    AcousticEventType,
    AudioFeatures,
    MFCCClassifier,
    TRAINING_DATA,
    ESC50_CATEGORY_MAP,
)


def test_acoustic_event_type_enum():
    """AcousticEventType has expected values."""
    assert AcousticEventType.GUNSHOT.value == "gunshot"
    assert AcousticEventType.VOICE.value == "voice"
    assert AcousticEventType.UNKNOWN.value == "unknown"


def test_audio_features_defaults():
    """AudioFeatures can be created with defaults."""
    f = AudioFeatures()
    assert f.rms_energy == 0.0
    assert f.mfcc is None


def test_acoustic_event_creation():
    """AcousticEvent can be created."""
    e = AcousticEvent(event_type=AcousticEventType.GUNSHOT, confidence=0.9)
    assert e.event_type == AcousticEventType.GUNSHOT
    assert e.confidence == 0.9


def test_training_data_exists():
    """TRAINING_DATA has entries for multiple classes."""
    assert len(TRAINING_DATA) > 20
    classes = set(t[0] for t in TRAINING_DATA)
    assert "gunshot" in classes
    assert "voice" in classes
    assert "vehicle" in classes


def test_esc50_category_map():
    """ESC50_CATEGORY_MAP maps to known acoustic types."""
    assert ESC50_CATEGORY_MAP["dog"] == "animal"
    assert ESC50_CATEGORY_MAP["siren"] == "siren"
    assert ESC50_CATEGORY_MAP["engine"] == "vehicle"


def test_mfcc_classifier_instantiation():
    """MFCCClassifier can be created."""
    c = MFCCClassifier(k=3)
    assert c.k == 3
    assert not c.is_trained


def test_mfcc_classifier_train():
    """MFCCClassifier trains on built-in data."""
    c = MFCCClassifier(k=5)
    c.train()
    assert c.is_trained


def test_mfcc_classifier_classify():
    """MFCCClassifier classifies features."""
    c = MFCCClassifier(k=5)
    c.train()

    # Gunshot-like features
    f = AudioFeatures(
        rms_energy=0.92,
        peak_amplitude=0.95,
        spectral_centroid=3500,
        zero_crossing_rate=0.15,
        spectral_bandwidth=4000,
        duration_ms=80,
        mfcc=[-40, 12, -5, 3, -2, 1, -1, 0.5, -0.3, 0.2, -0.1, 0.05, -0.02],
    )
    best_class, confidence, predictions = c.classify(f)
    assert isinstance(best_class, str)
    assert 0.0 <= confidence <= 1.0
    assert len(predictions) > 0


def test_acoustic_classifier_instantiation():
    """AcousticClassifier can be created."""
    ac = AcousticClassifier(enable_ml=True)
    assert ac is not None
    assert ac.ml_available


def test_acoustic_classifier_rule_based():
    """AcousticClassifier uses rule-based for non-MFCC features."""
    ac = AcousticClassifier(enable_ml=False)

    # Gunshot: high energy, short
    f = AudioFeatures(
        peak_amplitude=0.95,
        duration_ms=100,
        spectral_centroid=3500,
    )
    event = ac.classify(f)
    assert event.event_type == AcousticEventType.GUNSHOT


def test_acoustic_classifier_vehicle():
    """AcousticClassifier detects vehicles via rules."""
    ac = AcousticClassifier(enable_ml=False)

    f = AudioFeatures(
        spectral_centroid=200,
        duration_ms=5000,
        rms_energy=0.35,
        peak_amplitude=0.3,
    )
    event = ac.classify(f)
    assert event.event_type == AcousticEventType.VEHICLE


def test_acoustic_classifier_history():
    """AcousticClassifier tracks event history."""
    ac = AcousticClassifier(enable_ml=False)

    f = AudioFeatures(peak_amplitude=0.95, duration_ms=100, spectral_centroid=3500)
    ac.classify(f)
    ac.classify(f)

    events = ac.get_recent_events()
    assert len(events) == 2

    counts = ac.get_event_counts()
    assert sum(counts.values()) == 2
