# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""What a fired round leaves behind for the operator to see.

The geometry of a shot is already covered by ``tests/geo/test_hitscan.py``.
These tests cover the part that comes AFTER the ray is resolved: turning one
:class:`~tritium_lib.geo.hitscan.ShotResult` into a record the tactical map can
draw, and keeping a bounded history of them so an after-action review has
something to replay.
"""

from __future__ import annotations

import pytest

from tritium_lib.geo.hitscan import Muzzle, SphereTarget, resolve_shot
from tritium_lib.tracking.engagement import EngagementLog, ShotEvent


def _north_muzzle(up: float = 1.0) -> Muzzle:
    return Muzzle(east_m=0.0, north_m=0.0, up_m=up, heading_deg=0.0, elevation_deg=0.0)


# --- terminus -------------------------------------------------------------


def test_hit_terminus_is_the_impact_point():
    target = SphereTarget("dummy", east_m=0.0, north_m=10.0, up_m=1.0, radius_m=0.5)
    shot = resolve_shot(_north_muzzle(), [target], max_range_m=50.0)

    assert shot.hit
    east, north, up = ShotEvent.terminus_of(shot)
    assert north == pytest.approx(9.5, abs=1e-6)  # sphere SURFACE, not centre
    assert east == pytest.approx(0.0, abs=1e-6)
    assert up == pytest.approx(1.0, abs=1e-6)


def test_miss_terminus_is_the_max_range_point():
    """A miss still has to be drawable.

    A tracer that stops at nothing tells the operator nothing.  The round
    that missed travelled somewhere, and the honest terminus is the range
    gate along the aim -- not the origin, and not a silent None the renderer
    has to guess about.
    """
    target = SphereTarget("dummy", east_m=30.0, north_m=10.0, up_m=1.0, radius_m=0.5)
    shot = resolve_shot(_north_muzzle(), [target], max_range_m=25.0)

    assert not shot.hit
    east, north, up = ShotEvent.terminus_of(shot)
    assert north == pytest.approx(25.0, abs=1e-6)
    assert east == pytest.approx(0.0, abs=1e-6)


def test_out_of_range_hit_draws_to_the_gate_not_the_target():
    """Geometry hit, range gate refused: the tracer must not touch the body.

    Drawing to the impact point here would show the operator a round reaching
    a target that the range gate says it never reached -- a picture that
    contradicts the hit/miss verdict printed beside it.
    """
    target = SphereTarget("far", east_m=0.0, north_m=100.0, up_m=1.0, radius_m=0.5)
    shot = resolve_shot(_north_muzzle(), [target], max_range_m=10.0)

    assert not shot.hit
    _, north, _ = ShotEvent.terminus_of(shot)
    assert north == pytest.approx(10.0, abs=1e-6)


# --- the event ------------------------------------------------------------


def test_event_from_shot_carries_shooter_and_verdict():
    target = SphereTarget("dummy", east_m=0.0, north_m=10.0, up_m=1.0, radius_m=0.5)
    shot = resolve_shot(_north_muzzle(), [target], max_range_m=50.0)

    event = ShotEvent.from_shot(shot, shooter_id="go2_01", timestamp=1234.5)

    assert event.shooter_id == "go2_01"
    assert event.hit is True
    assert event.target_id == "dummy"
    assert event.timestamp == 1234.5
    assert event.range_m == pytest.approx(9.5, abs=1e-6)


def test_event_to_dict_is_json_safe_and_has_both_endpoints():
    """The renderer needs a LINE, so both ends must survive serialisation."""
    target = SphereTarget("dummy", east_m=0.0, north_m=10.0, up_m=1.0, radius_m=0.5)
    shot = resolve_shot(_north_muzzle(), [target], max_range_m=50.0)
    payload = ShotEvent.from_shot(shot, shooter_id="go2_01").to_dict()

    import json

    json.dumps(payload)  # must not raise

    assert payload["origin"] == [0.0, 0.0, 1.0]
    assert payload["terminus"][1] == pytest.approx(9.5, abs=1e-6)
    assert payload["hit"] is True
    assert payload["shot_id"]


def test_from_payload_accepts_a_wire_dict():
    """A connector sends ShotResult.to_dict() over the wire, not an object."""
    target = SphereTarget("dummy", east_m=0.0, north_m=10.0, up_m=1.0, radius_m=0.5)
    wire = resolve_shot(_north_muzzle(), [target], max_range_m=50.0).to_dict()

    event = ShotEvent.from_payload({**wire, "shooter_id": "go2_01"})

    assert event.hit is True
    assert event.shooter_id == "go2_01"
    assert event.terminus[1] == pytest.approx(9.5, abs=1e-6)


def test_from_payload_rejects_a_payload_with_no_muzzle():
    """No muzzle means no line to draw -- refuse rather than draw from zero."""
    with pytest.raises(ValueError):
        ShotEvent.from_payload({"hit": False})


def test_shot_ids_are_unique_across_events():
    target = SphereTarget("dummy", east_m=0.0, north_m=10.0, up_m=1.0, radius_m=0.5)
    shot = resolve_shot(_north_muzzle(), [target], max_range_m=50.0)
    ids = {ShotEvent.from_shot(shot).shot_id for _ in range(50)}
    assert len(ids) == 50


# --- the log --------------------------------------------------------------


def test_log_returns_most_recent_first():
    log = EngagementLog()
    for i in range(3):
        log.record({"hit": False, "max_range_m": 10.0,
                    "muzzle": _north_muzzle().to_dict(), "shooter_id": f"s{i}"})

    assert [e.shooter_id for e in log.recent()] == ["s2", "s1", "s0"]


def test_log_is_bounded_and_drops_the_oldest():
    """An engagement log on a long run must not grow without limit.

    A turret firing at 5 Hz for an hour is 18,000 records; an unbounded list
    of them in a long-lived Command Center process is a slow leak, and the
    operator only ever looks at the recent ones.
    """
    log = EngagementLog(max_events=5)
    for i in range(20):
        log.record({"hit": False, "max_range_m": 10.0,
                    "muzzle": _north_muzzle().to_dict(), "shooter_id": f"s{i}"})

    events = log.recent(limit=100)
    assert len(events) == 5
    assert [e.shooter_id for e in events] == ["s19", "s18", "s17", "s16", "s15"]


def test_log_limit_caps_the_response():
    log = EngagementLog()
    for i in range(10):
        log.record({"hit": False, "max_range_m": 10.0,
                    "muzzle": _north_muzzle().to_dict(), "shooter_id": f"s{i}"})

    assert len(log.recent(limit=3)) == 3


def test_log_since_filters_by_timestamp():
    log = EngagementLog()
    for i in range(5):
        log.record({"hit": False, "max_range_m": 10.0, "timestamp": float(i),
                    "muzzle": _north_muzzle().to_dict(), "shooter_id": f"s{i}"})

    assert [e.shooter_id for e in log.recent(since=2.0)] == ["s4", "s3"]


def test_log_counts_hits_and_misses_for_after_action():
    """Accuracy is the one number an after-action review always wants."""
    log = EngagementLog()
    muzzle = _north_muzzle().to_dict()
    for hit in (True, True, False, True):
        log.record({"hit": hit, "max_range_m": 10.0, "muzzle": muzzle,
                    "range_m": 5.0 if hit else None,
                    "impact": [0.0, 5.0, 1.0] if hit else None})

    stats = log.stats()
    assert stats["shots"] == 4
    assert stats["hits"] == 3
    assert stats["accuracy"] == pytest.approx(0.75)


def test_stats_on_an_empty_log_do_not_divide_by_zero():
    assert EngagementLog().stats() == {"shots": 0, "hits": 0, "accuracy": 0.0}


def test_record_rejects_a_malformed_payload():
    """A connector must not be able to poison the log with junk."""
    with pytest.raises(ValueError):
        EngagementLog().record({"hit": True})


def test_clear_empties_the_log():
    log = EngagementLog()
    log.record({"hit": False, "max_range_m": 10.0, "muzzle": _north_muzzle().to_dict()})
    log.clear()
    assert log.recent() == []
    assert log.stats()["shots"] == 0
