# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Stable-identity + capacity hardening for TargetTracker (2026-07).

Pins the contract added after the measured SC feedback cascade: a consumer
re-ingesting its own republished detections through update_from_detection
minted a fresh ``det_*`` track on EVERY call (no caller-supplied identity),
growing the tracker 0 -> 1,279 targets in 3.5 minutes (/api/targets 1.0 MB,
p50 latency 10.3 s).  Three defenses are pinned here:

1. ``detection_key`` / ``source_track_id`` — strict stable identity: the
   same key always updates the same live track; new keys mint; keys are
   never merged by proximity.
2. Byte-identity for unkeyed callers — no key means the historical
   proximity behavior, exactly.
3. ``max_targets`` — an opt-in hard membership cap on sensor-derived
   ingest, LOUD (WARNING log + ``tracker.capacity`` bus event + a
   ``cap_rejections`` counter), never a silent drop, and never applied to
   the operator's own fleet (simulation / robot pose).
"""

import logging

import pytest

from tritium_lib.tracking.target_tracker import TargetTracker


def _det(cls="person", conf=0.8, x=5.0, y=10.0, **extra):
    d = {"class_name": cls, "confidence": conf, "center_x": x, "center_y": y}
    d.update(extra)
    return d


class _BusStub:
    """Minimal EventBus stand-in — records every publish call."""

    def __init__(self):
        self.events = []

    def publish(self, topic, data=None, source=""):
        self.events.append((topic, data))


# ---------------------------------------------------------------------------
# 1. Stable identity: the detection_key contract
# ---------------------------------------------------------------------------

class TestDetectionKey:
    def test_same_key_twice_updates_one_track(self):
        tr = TargetTracker()
        t1 = tr.update_from_detection(_det(x=100.0, y=100.0), detection_key="camA:slot7")
        t2 = tr.update_from_detection(_det(x=250.0, y=-40.0), detection_key="camA:slot7")
        assert t1 == t2
        targets = tr.get_all()
        assert len(targets) == 1
        assert targets[0].signal_count == 2
        # The keyed update moved the one track — no ghost left behind.
        assert targets[0].position == (250.0, -40.0)

    def test_key_from_payload_detection_key(self):
        tr = TargetTracker()
        t1 = tr.update_from_detection(_det(x=100.0, y=100.0, detection_key="k1"))
        t2 = tr.update_from_detection(_det(x=400.0, y=400.0, detection_key="k1"))
        assert t1 == t2
        assert len(tr.get_all()) == 1

    def test_key_from_payload_source_track_id(self):
        tr = TargetTracker()
        t1 = tr.update_from_detection(_det(x=100.0, y=100.0, source_track_id="up_9"))
        t2 = tr.update_from_detection(_det(x=400.0, y=400.0, source_track_id="up_9"))
        assert t1 == t2
        assert len(tr.get_all()) == 1

    def test_different_keys_mint_distinct_tracks(self):
        tr = TargetTracker()
        t1 = tr.update_from_detection(_det(x=100.0, y=100.0), detection_key="k1")
        t2 = tr.update_from_detection(_det(x=500.0, y=500.0), detection_key="k2")
        assert t1 != t2
        assert len(tr.get_all()) == 2

    def test_distinct_keys_never_merge_even_colocated(self):
        """Supplying a key ASSERTS identity — two keys at the same spot are
        two entities (two people shoulder-to-shoulder in a crowd), and the
        keyed path must not proximity-collapse them."""
        tr = TargetTracker()
        t1 = tr.update_from_detection(_det(x=100.0, y=100.0), detection_key="person_a")
        t2 = tr.update_from_detection(_det(x=100.0, y=100.0), detection_key="person_b")
        assert t1 != t2
        assert len(tr.get_all()) == 2

    def test_key_survives_class_drift(self):
        """A key whose detector class flip-flops (person -> car) keeps
        updating the ONE track — identity outranks the class gate."""
        tr = TargetTracker()
        t1 = tr.update_from_detection(_det(cls="person", x=100.0, y=100.0), detection_key="k1")
        t2 = tr.update_from_detection(_det(cls="car", x=101.0, y=100.0), detection_key="k1")
        assert t1 == t2
        assert len(tr.get_all()) == 1

    def test_pruned_track_key_mints_fresh_never_resurrects(self):
        tr = TargetTracker()
        t1 = tr.update_from_detection(_det(x=100.0, y=100.0), detection_key="k1")
        tr.get_target(t1).last_seen -= (TargetTracker.STALE_TIMEOUT + 1.0)
        tr.get_all()  # prunes the stale vision track
        assert tr.get_target(t1) is None
        t2 = tr.update_from_detection(_det(x=100.0, y=100.0), detection_key="k1")
        assert t2 != t1
        assert len(tr.get_all()) == 1

    def test_keyed_below_confidence_gate_still_rejected(self):
        tr = TargetTracker()
        assert tr.update_from_detection(_det(conf=0.3), detection_key="k1") is None
        assert tr.get_all() == []

    def test_camera_bridge_forwards_key(self):
        tr = TargetTracker()
        fn = lambda lat, lng, alt=0.0: (100.0, 100.0, 0.0)
        t1 = tr.update_from_camera_detection(
            {"label": "person", "confidence": 0.9, "bbox": {"x": 0.2, "y": 0.5},
             "source_track_id": "bt_3"},
            camera_lat=0.0, camera_lng=0.0, latlng_to_local_fn=fn,
        )
        t2 = tr.update_from_camera_detection(
            {"label": "person", "confidence": 0.9, "bbox": {"x": 0.9, "y": 0.5},
             "source_track_id": "bt_3"},
            camera_lat=0.0, camera_lng=0.0, latlng_to_local_fn=fn,
        )
        assert t1 == t2
        assert len(tr.get_all()) == 1


# ---------------------------------------------------------------------------
# 2. Byte-identity: no key == the historical behavior, exactly
# ---------------------------------------------------------------------------

class TestUnkeyedBehaviorPinned:
    def test_unkeyed_id_sequence_pinned(self):
        """The exact id sequence today's (pre-hardening) code produces for
        this script.  Any drift in counter, id format, proximity matching,
        or the confidence gate breaks this test."""
        tr = TargetTracker()
        ids = [
            tr.update_from_detection(_det(cls="person", x=5.0, y=10.0)),
            tr.update_from_detection(_det(cls="person", x=5.5, y=10.0)),   # within radius -> match
            tr.update_from_detection(_det(cls="person", x=100.0, y=100.0)),  # far -> mint
            tr.update_from_detection(_det(cls="car", x=5.5, y=10.0)),      # class gate -> mint
            tr.update_from_detection(_det(cls="person", conf=0.3)),        # gate -> None
        ]
        assert ids == ["det_person_1", "det_person_1", "det_person_2", "det_car_3", None]
        assert len(tr.get_all()) == 3

    def test_unkeyed_track_fields_pinned(self):
        tr = TargetTracker()
        tid = tr.update_from_detection(_det(cls="person", conf=0.8, x=5.0, y=10.0))
        t = tr.get_target(tid)
        assert t.target_id == "det_person_1"
        assert t.name == "Person #1"
        assert t.alliance == "unknown"      # vision carries no IFF — pinned
        assert t.asset_type == "person"
        assert t.source == "yolo"
        assert t.position_source == "yolo"
        assert t.position == (5.0, 10.0)
        assert t.signal_count == 1
        assert t.position_confidence == 0.1
        assert t.classification == "person"
        assert t.classification_confidence == 0.8
        assert t.confirming_sources == {"yolo"}

    def test_unkeyed_match_updates_not_mints(self):
        tr = TargetTracker()
        tid = tr.update_from_detection(_det(x=5.0, y=10.0))
        for i in range(5):
            assert tr.update_from_detection(_det(x=5.0 + 0.01 * i, y=10.0)) == tid
        t = tr.get_target(tid)
        assert t.signal_count == 6
        assert len(tr.get_all()) == 1

    def test_no_cap_by_default(self):
        tr = TargetTracker()
        for i in range(100):
            tr.update_from_detection(_det(x=100.0 + 10.0 * i, y=500.0))
        assert len(tr.get_all()) == 100
        assert tr.cap_rejections == 0
        assert tr.max_targets is None


# ---------------------------------------------------------------------------
# 3. The adversarial republish-echo loop — THE sc cascade, pinned
# ---------------------------------------------------------------------------

class TestRepublishEchoLoop:
    def test_keyed_echo_plateaus(self):
        """Feed the tracker its own output repeatedly, stamped with each
        track's own id as source_track_id (exactly what a republishing
        consumer should now do).  Positions drift 14 m per pass — far
        outside the proximity radius — so without the key this is the
        cascade shape: unbounded minting.  With the key it PLATEAUS."""
        tr = TargetTracker()
        seeds = [
            tr.update_from_detection(_det(x=100.0 + 50.0 * i, y=200.0))
            for i in range(5)
        ]
        assert len(tr.get_all()) == 5

        for _ in range(50):
            for t in list(tr.get_all()):
                echo = _det(
                    cls=t.classification,
                    x=t.position[0] + 10.0,
                    y=t.position[1] + 10.0,
                )
                echo["source_track_id"] = t.target_id
                tr.update_from_detection(echo)

        targets = tr.get_all()
        assert len(targets) == 5, (
            f"republish echo grew the tracker to {len(targets)} tracks — "
            "the 0->1,279 cascade shape is back"
        )
        assert {t.target_id for t in targets} == set(seeds)
        # Every echo landed as an update on its own track.
        assert all(t.signal_count == 51 for t in targets)

    def test_external_stable_keys_echo_plateaus(self):
        """Same loop but keyed by an upstream id that is NOT one of our
        track ids (a ByteTrack/radar id).  First pass aliases each key to
        a track; every later pass updates through the alias."""
        tr = TargetTracker()
        for i in range(5):
            tr.update_from_detection(
                _det(x=100.0 + 50.0 * i, y=200.0), detection_key=f"upstream_{i}"
            )
        for _ in range(30):
            for i in range(5):
                tr.update_from_detection(
                    _det(x=100.0 + 50.0 * i + 10.0, y=210.0),
                    detection_key=f"upstream_{i}",
                )
        assert len(tr.get_all()) == 5

    def test_unkeyed_echo_grows_and_cap_is_the_last_line(self):
        """The UNKEYED echo still grows (documented hazard — republishing
        callers must supply keys); max_targets is the lib-side last line
        that clamps it, loudly."""
        tr = TargetTracker(max_targets=20)
        for i in range(5):
            tr.update_from_detection(_det(x=100.0 + 50.0 * i, y=200.0))

        for _ in range(20):
            for t in list(tr.get_all()):
                tr.update_from_detection(
                    _det(cls=t.classification,
                         x=t.position[0] + 10.0, y=t.position[1] + 10.0)
                )

        assert len(tr.get_all()) == 20      # clamped at the cap, not 100+
        assert tr.cap_rejections > 0        # and it was NOT silent


# ---------------------------------------------------------------------------
# 4. The hard cap: explicit, configurable, LOUD
# ---------------------------------------------------------------------------

class TestHardCap:
    def test_cap_refuses_new_but_updates_existing(self):
        tr = TargetTracker(max_targets=2)
        t1 = tr.update_from_detection(_det(x=100.0, y=100.0))
        t2 = tr.update_from_detection(_det(x=300.0, y=100.0))
        assert t1 and t2
        assert tr.update_from_detection(_det(x=500.0, y=100.0)) is None
        assert tr.cap_rejections == 1
        # Updates to existing tracks flow freely at cap.
        assert tr.update_from_detection(_det(x=100.5, y=100.0)) == t1
        assert tr.get_target(t1).signal_count == 2
        assert len(tr.get_all()) == 2

    def test_cap_is_loud_log_and_bus_event(self, caplog):
        bus = _BusStub()
        tr = TargetTracker(event_bus=bus, max_targets=1)
        tr.update_from_detection(_det(x=100.0, y=100.0))
        with caplog.at_level(logging.WARNING):
            assert tr.update_from_detection(_det(x=500.0, y=100.0)) is None
        assert any("capacity" in r.message.lower() for r in caplog.records)
        cap_events = [(t, d) for t, d in bus.events if t == "tracker.capacity"]
        assert len(cap_events) == 1
        payload = cap_events[0][1]
        assert payload["max_targets"] == 1
        assert payload["active_targets"] == 1
        assert payload["rejected_total"] == 1
        assert payload["source"] == "yolo"

    def test_cap_alarm_rate_limited_but_counter_exact(self):
        bus = _BusStub()
        tr = TargetTracker(event_bus=bus, max_targets=1)
        tr.update_from_detection(_det(x=100.0, y=100.0))
        for i in range(10):
            tr.update_from_detection(_det(x=500.0 + 10.0 * i, y=100.0))
        # One alarm within the rate-limit window, but EVERY refusal counted.
        assert len([1 for t, _ in bus.events if t == "tracker.capacity"]) == 1
        assert tr.cap_rejections == 10

    def test_cap_covers_ble_mesh_acoustic(self):
        tr = TargetTracker(max_targets=1, ble_classifier=False)
        tr.update_from_detection(_det(x=100.0, y=100.0))
        tr.update_from_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -50})
        tr.update_from_mesh({"target_id": "mesh_n1", "name": "n1",
                             "position": {"x": 5.0, "y": 5.0}})
        tr.update_from_acoustic({"event_type": "gunshot", "target_id": "x1",
                                 "position": {"x": 1.0, "y": 2.0}, "confidence": 0.5})
        assert len(tr.get_all()) == 1
        assert tr.cap_rejections == 3

    def test_ble_update_still_flows_at_cap(self):
        tr = TargetTracker(max_targets=1, ble_classifier=False)
        tr.update_from_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -50})
        assert len(tr.get_all()) == 1
        tr.update_from_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -40})
        t = tr.get_all()[0]
        assert t.signal_count == 2
        assert tr.cap_rejections == 0

    def test_own_fleet_exempt_from_cap(self):
        """A full tracker must never blind the operator to their own
        assets — simulation and robot-pose ingest bypass the cap."""
        tr = TargetTracker(max_targets=1)
        tr.update_from_detection(_det(x=100.0, y=100.0))
        tr.update_from_simulation({
            "target_id": "rover_01", "name": "Alpha",
            "alliance": "friendly", "asset_type": "rover",
            "position": {"x": 1.0, "y": 2.0},
        })
        assert tr.update_from_robot_pose({
            "target_id": "dog_01", "position": {"x": 3.0, "y": 4.0},
            "heading": 90.0,
        }) == "dog_01"
        assert tr.get_target("rover_01") is not None
        assert tr.get_target("dog_01") is not None
        assert len(tr.get_all()) == 3


# ---------------------------------------------------------------------------
# 5. Alias-map hygiene — the fix must not itself leak
# ---------------------------------------------------------------------------

class TestAliasHygiene:
    def test_alias_map_purged_with_pruned_tracks(self):
        tr = TargetTracker()
        for i in range(10):
            tr.update_from_detection(
                _det(x=100.0 + 50.0 * i, y=100.0), detection_key=f"k{i}"
            )
        assert len(tr._detection_keys) == 10
        for t in tr.get_all():
            t.last_seen -= (TargetTracker.STALE_TIMEOUT + 1.0)
        tr.get_all()  # prune
        assert tr._detection_keys == {}

    def test_alias_map_purged_on_remove(self):
        tr = TargetTracker()
        tid = tr.update_from_detection(_det(x=100.0, y=100.0), detection_key="k1")
        assert tr.remove(tid)
        assert tr._detection_keys == {}

    def test_alias_map_lru_capped(self):
        tr = TargetTracker()
        tid = tr.update_from_detection(_det(x=100.0, y=100.0), detection_key="k_first")
        limit = TargetTracker.MAX_DETECTION_KEY_ALIASES
        # Flood with distinct keys that all resolve (by fresh mints far apart
        # is too slow) — alias directly at the internal map to prove the bound.
        with tr._lock:
            for i in range(limit + 100):
                tr._record_detection_key_locked(f"flood_{i}", tid)
            assert len(tr._detection_keys) == limit

    def test_self_alias_not_stored(self):
        """A caller echoing our own det_* id back resolves directly against
        the registry — no alias entry needed or stored."""
        tr = TargetTracker()
        tid = tr.update_from_detection(_det(x=100.0, y=100.0))
        t2 = tr.update_from_detection(
            _det(x=110.0, y=110.0), detection_key=tid
        )
        assert t2 == tid
        assert tr._detection_keys == {}
