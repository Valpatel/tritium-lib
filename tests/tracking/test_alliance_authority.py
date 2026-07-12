# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Alliance-authority precedence tests (2026-07-11 ruling).

The TargetTracker is the ONE authority for a target's effective alliance.
Precedence, highest first:

    1. operator tag        (set_operator_alliance — pinned)
    2. declared telemetry  ("alliance" key in an update_from_simulation frame)
    3. creation default    (whatever the ingest stamped at first sight)

Before this ruling, update_from_simulation NEVER updated alliance on the
existing-target branch — a robot re-declaring its alliance mid-run was
silently dropped, while SC's ws.py re-resolved alliance from raw wire data
every frame and the tag route wrote the tracker directly: three
uncoordinated writers, live map vs REST vs CoT permanently disagreeing.
"""

from tritium_lib.tracking.target_tracker import (
    VALID_ALLIANCES,
    TargetTracker,
)


def _spawn(tracker: TargetTracker, alliance: str = "friendly") -> str:
    tracker.update_from_simulation({
        "target_id": "unit_01",
        "name": "Unit One",
        "alliance": alliance,
        "asset_type": "rover",
        "position": {"x": 1.0, "y": 2.0},
    })
    return "unit_01"


class TestDeclaredTelemetryTier:
    """Tier 2: the tracker follows a declared mid-run alliance change."""

    def test_mid_run_declared_change_applies(self):
        """A unit that re-declares its alliance mid-run must not be dropped."""
        tracker = TargetTracker()
        tid = _spawn(tracker, alliance="friendly")

        tracker.update_from_simulation({
            "target_id": tid,
            "position": {"x": 3.0, "y": 4.0},
            "alliance": "hostile",
        })

        assert tracker.get_target(tid).alliance == "hostile"
        assert tid in [t.target_id for t in tracker.get_hostiles()]
        assert tid not in [t.target_id for t in tracker.get_friendlies()]

    def test_absent_key_is_no_opinion(self):
        """A frame without the alliance key never resets a known alliance."""
        tracker = TargetTracker()
        tid = _spawn(tracker, alliance="hostile")

        tracker.update_from_simulation({
            "target_id": tid,
            "position": {"x": 3.0, "y": 4.0},
        })

        assert tracker.get_target(tid).alliance == "hostile"

    def test_junk_value_never_clobbers(self):
        """Values outside VALID_ALLIANCES are dropped, not applied."""
        tracker = TargetTracker()
        tid = _spawn(tracker, alliance="friendly")

        tracker.update_from_simulation({
            "target_id": tid,
            "position": {"x": 3.0, "y": 4.0},
            "alliance": "zerg",
        })
        tracker.update_from_simulation({
            "target_id": tid,
            "position": {"x": 3.0, "y": 4.0},
            "alliance": None,
        })

        assert tracker.get_target(tid).alliance == "friendly"

    def test_declared_change_bumps_version_for_etag(self):
        """An alliance flip must invalidate /api/targets ETag caches."""
        tracker = TargetTracker()
        tid = _spawn(tracker, alliance="friendly")
        v0 = tracker.version

        # Position-only frame: no bump (streaming state, not identity).
        tracker.update_from_simulation({"target_id": tid, "position": {"x": 9.0, "y": 9.0}})
        assert tracker.version == v0

        tracker.update_from_simulation({
            "target_id": tid,
            "position": {"x": 3.0, "y": 4.0},
            "alliance": "hostile",
        })
        assert tracker.version == v0 + 1

        # Same alliance re-declared: no spurious bump.
        tracker.update_from_simulation({
            "target_id": tid,
            "position": {"x": 3.5, "y": 4.5},
            "alliance": "hostile",
        })
        assert tracker.version == v0 + 1


class TestOperatorTier:
    """Tier 1: an explicit operator tag pins the alliance."""

    def test_operator_tag_applies_and_pins(self):
        tracker = TargetTracker()
        tid = _spawn(tracker, alliance="friendly")

        t = tracker.set_operator_alliance(tid, "hostile")

        assert t is not None
        assert t.alliance == "hostile"
        assert t.alliance_source == "operator"

    def test_operator_tag_outranks_declared_telemetry(self):
        """The dog keeps declaring friendly every frame; the human said hostile."""
        tracker = TargetTracker()
        tid = _spawn(tracker, alliance="friendly")
        tracker.set_operator_alliance(tid, "hostile")

        for _ in range(5):
            tracker.update_from_simulation({
                "target_id": tid,
                "position": {"x": 3.0, "y": 4.0},
                "alliance": "friendly",
            })

        target = tracker.get_target(tid)
        assert target.alliance == "hostile"
        assert target.alliance_source == "operator"

    def test_operator_retag_wins_over_previous_tag(self):
        tracker = TargetTracker()
        tid = _spawn(tracker, alliance="friendly")
        tracker.set_operator_alliance(tid, "hostile")
        tracker.set_operator_alliance(tid, "vip")

        assert tracker.get_target(tid).alliance == "vip"

    def test_invalid_operator_value_rejected(self):
        tracker = TargetTracker()
        tid = _spawn(tracker, alliance="friendly")

        assert tracker.set_operator_alliance(tid, "protoss") is None
        assert tracker.get_target(tid).alliance == "friendly"
        assert tracker.get_target(tid).alliance_source == "auto"

    def test_unknown_target_returns_none(self):
        tracker = TargetTracker()
        assert tracker.set_operator_alliance("ghost_99", "hostile") is None

    def test_operator_tag_bumps_version_for_etag(self):
        tracker = TargetTracker()
        tid = _spawn(tracker, alliance="friendly")
        v0 = tracker.version

        tracker.set_operator_alliance(tid, "hostile")

        assert tracker.version == v0 + 1


class TestSerializationAndVocabulary:
    """The wire contract every consumer reads."""

    def test_to_dict_carries_alliance_source(self):
        tracker = TargetTracker()
        tid = _spawn(tracker)
        assert tracker.get_target(tid).to_dict()["alliance_source"] == "auto"

        tracker.set_operator_alliance(tid, "hostile")
        d = tracker.get_target(tid).to_dict()
        assert d["alliance"] == "hostile"
        assert d["alliance_source"] == "operator"

    def test_valid_alliances_vocabulary(self):
        """SC routes validate against this exact set — keep it canonical."""
        assert VALID_ALLIANCES == frozenset(
            {"friendly", "hostile", "neutral", "unknown", "vip"}
        )


class TestConvergenceStory:
    """The full before/after story: mid-run flip + operator tag both converge."""

    def test_mid_run_flip_then_operator_tag(self):
        """Declared flip tracks the wire; operator tag then freezes it."""
        tracker = TargetTracker()
        tid = _spawn(tracker, alliance="friendly")

        # Robot declares hostile mid-run -> tracker follows (tier 2).
        tracker.update_from_simulation({
            "target_id": tid,
            "position": {"x": 5.0, "y": 5.0},
            "alliance": "hostile",
        })
        assert tracker.get_target(tid).alliance == "hostile"

        # Operator overrides to friendly (tier 1) -> wire can't flip it back.
        tracker.set_operator_alliance(tid, "friendly")
        tracker.update_from_simulation({
            "target_id": tid,
            "position": {"x": 6.0, "y": 6.0},
            "alliance": "hostile",
        })

        target = tracker.get_target(tid)
        assert target.alliance == "friendly"
        # Every read surface sees the same value: direct, list, dict.
        assert tid in [t.target_id for t in tracker.get_friendlies()]
        assert target.to_dict()["alliance"] == "friendly"
