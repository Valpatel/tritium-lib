# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.dossier."""

import threading
import pytest

pytestmark = pytest.mark.unit

from tritium_lib.tracking.dossier import DossierStore, TargetDossier


# --- TargetDossier ---

def test_dossier_default_uuid():
    d = TargetDossier()
    assert len(d.uuid) == 36  # UUID4 format
    assert d.signal_ids == []
    assert d.confidence == 0.0


def test_dossier_add_signal():
    d = TargetDossier()
    d.add_signal("ble_aa:bb:cc", "ble")
    assert "ble_aa:bb:cc" in d.signal_ids
    assert "ble" in d.sources


def test_dossier_add_signal_deduplication():
    d = TargetDossier()
    d.add_signal("ble_aa:bb:cc", "ble")
    d.add_signal("ble_aa:bb:cc", "ble")
    assert d.signal_ids.count("ble_aa:bb:cc") == 1
    assert d.sources.count("ble") == 1


def test_dossier_has_signal():
    d = TargetDossier()
    d.add_signal("ble_aa:bb:cc", "ble")
    assert d.has_signal("ble_aa:bb:cc") is True
    assert d.has_signal("wifi_dd:ee:ff") is False


def test_dossier_to_dict():
    d = TargetDossier(confidence=0.85)
    d.add_signal("ble_aa:bb:cc", "ble")
    result = d.to_dict()
    assert result["uuid"] == d.uuid
    assert result["confidence"] == 0.85
    assert "ble_aa:bb:cc" in result["signal_ids"]


# --- DossierStore ---

def test_store_empty():
    store = DossierStore()
    assert store.count == 0
    assert store.get_all() == []


def test_store_create_new_dossier():
    store = DossierStore()
    dossier = store.create_or_update(
        "ble_aa:bb:cc", "ble", "det_person_1", "yolo", 0.9
    )
    assert store.count == 1
    assert dossier.has_signal("ble_aa:bb:cc")
    assert dossier.has_signal("det_person_1")
    assert dossier.confidence == 0.9


def test_store_find_by_signal():
    store = DossierStore()
    store.create_or_update("ble_aa:bb:cc", "ble", "det_person_1", "yolo", 0.9)
    found = store.find_by_signal("ble_aa:bb:cc")
    assert found is not None
    assert found.has_signal("det_person_1")


def test_store_find_by_uuid():
    store = DossierStore()
    dossier = store.create_or_update("sig_a", "ble", "sig_b", "yolo", 0.5)
    found = store.find_by_uuid(dossier.uuid)
    assert found is not None
    assert found.uuid == dossier.uuid


def test_store_find_by_signal_missing():
    store = DossierStore()
    assert store.find_by_signal("nonexistent") is None


def test_store_find_association():
    store = DossierStore()
    store.create_or_update("sig_a", "ble", "sig_b", "yolo", 0.8)
    assoc = store.find_association("sig_a", "sig_b")
    assert assoc is not None
    # Unrelated signals should return None
    assert store.find_association("sig_a", "sig_c") is None


def test_store_update_existing_same_dossier():
    store = DossierStore()
    d1 = store.create_or_update("sig_a", "ble", "sig_b", "yolo", 0.5)
    d2 = store.create_or_update("sig_a", "ble", "sig_b", "yolo", 0.7)
    assert d1.uuid == d2.uuid
    assert d2.confidence == 0.7
    assert d2.correlation_count == 2  # 1 initial + 1 update


def test_store_add_signal_to_existing():
    store = DossierStore()
    store.create_or_update("sig_a", "ble", "sig_b", "yolo", 0.5)
    d = store.create_or_update("sig_a", "ble", "sig_c", "wifi", 0.6)
    assert d.has_signal("sig_c")
    assert store.count == 1  # still one dossier


def test_store_merge_two_dossiers():
    store = DossierStore()
    d1 = store.create_or_update("sig_a", "ble", "sig_b", "yolo", 0.9)
    d2 = store.create_or_update("sig_c", "wifi", "sig_d", "camera", 0.3)
    assert store.count == 2
    # Now correlate across dossiers — higher confidence wins
    merged = store.create_or_update("sig_a", "ble", "sig_c", "wifi", 0.5)
    assert store.count == 1
    assert merged.has_signal("sig_a")
    assert merged.has_signal("sig_b")
    assert merged.has_signal("sig_c")
    assert merged.has_signal("sig_d")
    assert merged.confidence == 0.9  # kept higher


def test_store_clear():
    store = DossierStore()
    store.create_or_update("sig_a", "ble", "sig_b", "yolo", 0.5)
    store.clear()
    assert store.count == 0
    assert store.find_by_signal("sig_a") is None


def test_store_metadata():
    store = DossierStore()
    d = store.create_or_update(
        "sig_a", "ble", "sig_b", "yolo", 0.5,
        metadata={"tag": "suspect"}
    )
    assert d.metadata["tag"] == "suspect"
    # Update metadata
    store.create_or_update(
        "sig_a", "ble", "sig_b", "yolo", 0.6,
        metadata={"color": "red"}
    )
    assert d.metadata["tag"] == "suspect"
    assert d.metadata["color"] == "red"


def test_store_thread_safety():
    store = DossierStore()
    errors = []

    def writer(n):
        try:
            for i in range(20):
                store.create_or_update(
                    f"sig_{n}_{i}", "ble", f"det_{n}_{i}", "yolo", 0.5
                )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert store.count > 0
