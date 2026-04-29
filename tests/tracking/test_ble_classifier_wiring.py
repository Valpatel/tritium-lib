# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Gap-fix B-7: behavioral tests for BLE classifier wiring in TargetTracker.

Before this fix, every BLE-only target had ``classification_confidence == 0.0``
because :meth:`TargetTracker.update_from_ble` only honoured classification
fields the caller passed in the sighting dict.  In production, the BLE
scanner publishes raw sightings (mac/name/rssi only) so 0/24 BLE targets
ever ended up with non-zero classification confidence.

These tests assert that the tracker now invokes the bundled
``DeviceClassifier`` over whatever identity hints arrive in the sighting
and writes the resulting (device_type, confidence) onto the target.
"""

from tritium_lib.classifier import DeviceClassifier
from tritium_lib.tracking.target_tracker import TargetTracker


def _make_tracker():
    # Pass a fresh DeviceClassifier instance so the test does not depend
    # on the process-wide lazy singleton state (other tests may have
    # already faulted it in).
    return TargetTracker(ble_classifier=DeviceClassifier())


def test_apple_mac_prefix_yields_phone_classification():
    """Sighting with a known Apple OUI (AC:BC:32) and ``iPhone`` name
    should be classified as a phone with non-trivial confidence.
    """
    tracker = _make_tracker()
    tracker.update_from_ble({
        "mac": "AC:BC:32:11:22:33",
        "name": "iPhone 15",
        "rssi": -55,
    })
    target = tracker.get_target("ble_acbc3211 2233".replace(" ", ""))
    assert target is not None
    assert target.classification == "phone"
    assert target.classification_confidence >= 0.85
    # Asset type should also be upgraded from the generic "ble_device"
    assert target.asset_type == "phone"


def test_random_mac_with_iphone_name_classified_via_name_pattern():
    """Even without a recognizable OUI (randomized MAC), the advertised
    name alone is enough for the classifier to label it as a phone.
    """
    tracker = _make_tracker()
    # Locally-administered MAC (bit 1 of first octet set => "02") so the
    # OUI signal is suppressed; only the name pattern carries the day.
    tracker.update_from_ble({
        "mac": "02:11:22:33:44:55",
        "name": "iPhone 14 Pro",
        "rssi": -65,
    })
    target = tracker.get_target("ble_021122334455")
    assert target is not None
    assert target.classification == "phone"
    assert target.classification_confidence > 0.0


def test_unknown_device_does_not_lose_default_classification():
    """A sighting with no recognizable identity hints should still produce
    a target — classification stays at the default but the tracker must
    not crash.
    """
    tracker = _make_tracker()
    tracker.update_from_ble({
        "mac": "12:34:56:78:9A:BC",
        "name": "MysteryDevice",
        "rssi": -85,
    })
    target = tracker.get_target("ble_123456789abc")
    assert target is not None
    # Either the default ble_device falls back, or the name pattern DB
    # matches something — either way confidence must be a valid float.
    assert isinstance(target.classification_confidence, float)
    assert 0.0 <= target.classification_confidence <= 1.0


def test_explicit_classification_in_sighting_is_preserved():
    """If the upstream caller already classified the device, the tracker
    must NOT overwrite that with a derived classification.
    """
    tracker = _make_tracker()
    tracker.update_from_ble({
        "mac": "AC:BC:32:11:22:33",
        "name": "iPhone 15",
        "rssi": -55,
        "classification": "watch",
        "classification_confidence": 0.99,
    })
    target = tracker.get_target("ble_acbc32112233")
    assert target is not None
    assert target.classification == "watch"
    assert target.classification_confidence == 0.99


def test_disabled_classifier_leaves_classification_blank():
    """Passing ``ble_classifier=False`` disables auto-classification — used
    by tests/tools that want to assert the old behaviour or avoid loading
    the JSON databases.
    """
    tracker = TargetTracker(ble_classifier=False)
    tracker.update_from_ble({
        "mac": "AC:BC:32:11:22:33",
        "name": "iPhone 15",
        "rssi": -55,
    })
    target = tracker.get_target("ble_acbc32112233")
    assert target is not None
    # Falls back to the legacy default (asset_type) since classifier ran
    # nothing.  Confidence stays at 0.0 because no sighting hint provided
    # one.
    assert target.classification_confidence == 0.0


def test_subsequent_ble_update_can_upgrade_classification():
    """First sighting has a generic name; second sighting carries a
    richer name pattern — the tracker should adopt the higher-confidence
    classification on the second update.
    """
    tracker = _make_tracker()
    tracker.update_from_ble({
        "mac": "AC:BC:32:11:22:33",
        "name": "Unknown",
        "rssi": -75,
    })
    first = tracker.get_target("ble_acbc32112233")
    assert first is not None
    initial_conf = first.classification_confidence

    tracker.update_from_ble({
        "mac": "AC:BC:32:11:22:33",
        "name": "iPhone 15",
        "rssi": -55,
    })
    second = tracker.get_target("ble_acbc32112233")
    assert second is not None
    assert second.classification == "phone"
    assert second.classification_confidence >= initial_conf


def test_company_id_alone_yields_classification():
    """Some BLE scanners publish only mfg-specific data (company_id) with
    no name.  The classifier should still produce a sensible vote.
    """
    tracker = _make_tracker()
    tracker.update_from_ble({
        "mac": "12:34:56:78:9A:BC",
        "rssi": -70,
        # 76 = Apple Inc (0x004C) in the BLE company-id list.
        "company_id": 76,
    })
    target = tracker.get_target("ble_123456789abc")
    assert target is not None
    # Apple's company-id lookup will produce *some* device_type vote;
    # we only assert the wiring fired (confidence > 0).
    assert target.classification_confidence > 0.0
