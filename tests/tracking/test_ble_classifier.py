# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.ble_classifier."""

import pytest
from tritium_lib.tracking.ble_classifier import (
    BLEClassifier,
    BLEClassification,
    CLASSIFICATION_LEVELS,
    DEFAULT_SUSPICIOUS_RSSI,
)


class FakeEventBus:
    def __init__(self):
        self.events = []

    def publish(self, topic, data):
        self.events.append((topic, data))


# --- Classification levels ---

def test_known_device_classified_as_known():
    c = BLEClassifier(known_macs={"AA:BB:CC:DD:EE:FF"})
    result = c.classify("AA:BB:CC:DD:EE:FF", "Test", -50)
    assert result.level == "known"


def test_new_device_classified_as_new():
    c = BLEClassifier()
    result = c.classify("AA:BB:CC:DD:EE:FF", "Test", -80)
    assert result.level == "new"


def test_suspicious_strong_signal_unknown():
    c = BLEClassifier()
    result = c.classify("AA:BB:CC:DD:EE:FF", "", -30)
    assert result.level == "suspicious"


def test_previously_seen_becomes_unknown():
    c = BLEClassifier()
    c.classify("AA:BB:CC:DD:EE:FF", "Test", -80)  # first time -> new
    result = c.classify("AA:BB:CC:DD:EE:FF", "Test", -80)  # second -> unknown
    assert result.level == "unknown"


def test_previously_seen_strong_signal_suspicious():
    c = BLEClassifier()
    c.classify("AA:BB:CC:DD:EE:FF", "Test", -80)  # new
    result = c.classify("AA:BB:CC:DD:EE:FF", "Test", -30)  # strong -> suspicious
    assert result.level == "suspicious"


# --- MAC normalization ---

def test_mac_uppercased():
    c = BLEClassifier(known_macs={"aa:bb:cc:dd:ee:ff"})
    result = c.classify("aa:bb:cc:dd:ee:ff", "Test", -50)
    assert result.mac == "AA:BB:CC:DD:EE:FF"
    assert result.level == "known"


# --- Seen count ---

def test_seen_count_increments():
    c = BLEClassifier()
    c.classify("AA:BB:CC:DD:EE:FF", "Test", -80)
    c.classify("AA:BB:CC:DD:EE:FF", "Test", -75)
    c.classify("AA:BB:CC:DD:EE:FF", "Test", -70)
    result = c.get_classifications()["AA:BB:CC:DD:EE:FF"]
    assert result.seen_count == 3


# --- Known MAC management ---

def test_add_known_reclassifies():
    c = BLEClassifier()
    c.classify("AA:BB:CC:DD:EE:FF", "Test", -80)
    c.add_known("AA:BB:CC:DD:EE:FF")
    result = c.get_classifications()["AA:BB:CC:DD:EE:FF"]
    assert result.level == "known"


def test_remove_known():
    c = BLEClassifier(known_macs={"AA:BB:CC:DD:EE:FF"})
    assert c.remove_known("AA:BB:CC:DD:EE:FF") is True
    assert c.remove_known("AA:BB:CC:DD:EE:FF") is False


def test_get_known_macs():
    c = BLEClassifier(known_macs={"AA:BB:CC:DD:EE:FF", "11:22:33:44:55:66"})
    macs = c.get_known_macs()
    assert len(macs) == 2
    assert "AA:BB:CC:DD:EE:FF" in macs


# --- Event publishing ---

def test_new_device_publishes_events():
    bus = FakeEventBus()
    c = BLEClassifier(event_bus=bus)
    c.classify("AA:BB:CC:DD:EE:FF", "NewPhone", -80)
    topics = [e[0] for e in bus.events]
    assert "ble:new_device" in topics
    assert "ble:first_seen" in topics


def test_suspicious_device_publishes_event():
    bus = FakeEventBus()
    c = BLEClassifier(event_bus=bus)
    c.classify("AA:BB:CC:DD:EE:FF", "", -80)  # first: new
    bus.events.clear()
    c.classify("AA:BB:CC:DD:EE:FF", "", -30)  # strong signal: suspicious
    topics = [e[0] for e in bus.events]
    assert "ble:suspicious_device" in topics


def test_known_device_no_alert():
    bus = FakeEventBus()
    c = BLEClassifier(event_bus=bus, known_macs={"AA:BB:CC:DD:EE:FF"})
    c.classify("AA:BB:CC:DD:EE:FF", "Known", -30)
    assert len(bus.events) == 0


def test_no_event_bus_no_crash():
    c = BLEClassifier(event_bus=None)
    result = c.classify("AA:BB:CC:DD:EE:FF", "Test", -80)
    assert result.level == "new"


# --- Training store callback ---

def test_training_store_callback():
    logged = []

    class FakeStore:
        def log_classification(self, **kwargs):
            logged.append(kwargs)

    c = BLEClassifier(training_store_fn=lambda: FakeStore())
    c.classify("AA:BB:CC:DD:EE:FF", "Test", -50)
    assert len(logged) == 1
    assert logged[0]["source"] == "ble_classifier"


def test_training_store_none_no_crash():
    c = BLEClassifier(training_store_fn=None)
    c.classify("AA:BB:CC:DD:EE:FF", "Test", -50)  # should not crash


# --- Filter and clear ---

def test_get_classifications_by_level():
    c = BLEClassifier(known_macs={"11:22:33:44:55:66"})
    c.classify("11:22:33:44:55:66", "Known", -50)
    c.classify("AA:BB:CC:DD:EE:FF", "New", -80)
    known = c.get_classifications_by_level("known")
    assert len(known) == 1
    assert known[0].mac == "11:22:33:44:55:66"


def test_clear_resets_state():
    c = BLEClassifier()
    c.classify("AA:BB:CC:DD:EE:FF", "Test", -80)
    c.clear()
    assert len(c.get_classifications()) == 0
    # After clear, same MAC is "new" again
    result = c.classify("AA:BB:CC:DD:EE:FF", "Test", -80)
    assert result.level == "new"


# --- Classification levels constant ---

def test_classification_levels():
    assert "known" in CLASSIFICATION_LEVELS
    assert "suspicious" in CLASSIFICATION_LEVELS
    assert len(CLASSIFICATION_LEVELS) == 4
