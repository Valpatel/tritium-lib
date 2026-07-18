"""Tests for the live command link — the seam a driven body listens on.

Written before the implementation.  The properties that matter here are the
ones a happy-path test never reaches: a garbage packet must not count as
liveness, a reordered datagram must not rewind the body, and a *restarted*
sender must not be locked out forever by a sequence counter it no longer
remembers.
"""

import json

import pytest

from tritium_lib.control import TwistCommand
from tritium_lib.control.command_link import CommandLink, CommandLimits


def frame(seq, linear=0.0, angular=0.0):
    return json.dumps(
        {"cmd": "twist", "seq": seq, "linear_mps": linear, "angular_rps": angular}
    ).encode()


@pytest.fixture
def link():
    return CommandLink(
        limits=CommandLimits(max_linear_mps=1.0, max_angular_rps=2.0),
        timeout_s=0.5,
    )


# ---------------------------------------------------------------- basic flow


def test_no_input_yet_commands_a_stop(link):
    assert link.poll(0.0) == TwistCommand.stop()


def test_accepted_frame_is_the_commanded_twist(link):
    assert link.ingest(frame(1, 0.4, 0.2), now_s=0.0) is True
    assert link.poll(0.0) == TwistCommand(linear_mps=0.4, angular_rps=0.2)


def test_held_between_frames(link):
    link.ingest(frame(1, 0.4, 0.0), now_s=0.0)
    # Within the timeout the last command still stands — a control loop runs
    # far faster than the link delivers.
    assert link.poll(0.4).linear_mps == pytest.approx(0.4)


# ------------------------------------------------------------------- safety


def test_silence_trips_the_watchdog_to_a_stop(link):
    link.ingest(frame(1, 0.9, 0.0), now_s=0.0)
    assert link.poll(0.6) == TwistCommand.stop()


def test_watchdog_stays_tripped_until_a_fresh_frame(link):
    """Un-tripping on the clock alone would let a dead link stutter the body."""
    link.ingest(frame(1, 0.9, 0.0), now_s=0.0)
    assert link.poll(0.6) == TwistCommand.stop()
    # No new frame; more time passes. Still stopped.
    assert link.poll(0.7) == TwistCommand.stop()
    link.ingest(frame(2, 0.5, 0.0), now_s=0.8)
    assert link.poll(0.8).linear_mps == pytest.approx(0.5)


def test_garbage_is_rejected_and_does_not_count_as_liveness(link):
    """The subtle one: a malformed packet must not feed the watchdog.

    If corrupt traffic refreshed the deadline, a sender emitting nothing but
    garbage would hold the body at its last good command indefinitely — the
    exact failure the watchdog exists to prevent.
    """
    link.ingest(frame(1, 0.9, 0.0), now_s=0.0)
    for junk in (b"", b"not json", b"{}", b'{"cmd":"twist"}', b'{"cmd":"fire","seq":9}'):
        assert link.ingest(junk, now_s=0.4) is False
    assert link.poll(0.6) == TwistCommand.stop()
    assert link.rejected == 5


def test_non_finite_values_are_rejected(link):
    """NaN through a stride mixer poisons every joint target downstream."""
    bad = json.dumps(
        {"cmd": "twist", "seq": 1, "linear_mps": float("nan"), "angular_rps": 0.0}
    ).encode()
    assert link.ingest(bad, now_s=0.0) is False
    assert link.poll(0.0) == TwistCommand.stop()


def test_values_are_clamped_to_the_configured_limits(link):
    """A sender is not trusted to respect the body's envelope."""
    link.ingest(frame(1, 99.0, -99.0), now_s=0.0)
    assert link.poll(0.0) == TwistCommand(linear_mps=1.0, angular_rps=-2.0)


# --------------------------------------------------------------- ordering


def test_stale_sequence_is_dropped(link):
    """UDP reorders. An older command arriving late must not rewind the body."""
    link.ingest(frame(5, 0.5, 0.0), now_s=0.0)
    assert link.ingest(frame(4, -0.5, 0.0), now_s=0.1) is False
    assert link.poll(0.1).linear_mps == pytest.approx(0.5)


def test_duplicate_sequence_is_dropped(link):
    link.ingest(frame(5, 0.5, 0.0), now_s=0.0)
    assert link.ingest(frame(5, -0.5, 0.0), now_s=0.1) is False


def test_a_dropped_stale_frame_does_not_refresh_the_watchdog(link):
    """Reordered traffic is not proof the sender is still alive *now*."""
    link.ingest(frame(5, 0.5, 0.0), now_s=0.0)
    link.ingest(frame(4, 0.5, 0.0), now_s=0.4)
    assert link.poll(0.6) == TwistCommand.stop()


def test_a_restarted_sender_is_adopted_not_locked_out(link):
    """The bug that bricks a body: a sender restarts, its counter resets to 0,
    and a strictly-monotonic receiver drops every packet forever.

    A large *backwards* jump is a new session, not a reordered datagram — real
    reordering spans a handful of packets, never thousands.
    """
    link.ingest(frame(5000, 0.5, 0.0), now_s=0.0)
    assert link.ingest(frame(0, -0.3, 0.0), now_s=1.0) is True
    assert link.poll(1.0).linear_mps == pytest.approx(-0.3)
    # And the new session's ordering is enforced from its own baseline.
    assert link.ingest(frame(1, 0.2, 0.0), now_s=1.1) is True


def test_sequence_is_optional(link):
    """A sender that omits seq gets liveness and clamping, just no ordering."""
    payload = json.dumps({"cmd": "twist", "linear_mps": 0.3, "angular_rps": 0.0})
    assert link.ingest(payload.encode(), now_s=0.0) is True
    assert link.poll(0.0).linear_mps == pytest.approx(0.3)


# ---------------------------------------------------------------- accounting


def test_counters_track_accepted_and_rejected(link):
    link.ingest(frame(1), now_s=0.0)
    link.ingest(frame(2), now_s=0.1)
    link.ingest(b"junk", now_s=0.2)
    link.ingest(frame(1), now_s=0.3)  # stale
    assert link.accepted == 2
    assert link.rejected == 2


def test_stop_command_is_accepted_as_a_real_command(link):
    """An explicit stop is a command, not an absence of one."""
    link.ingest(frame(1, 0.8, 0.0), now_s=0.0)
    assert link.ingest(frame(2, 0.0, 0.0), now_s=0.1) is True
    assert link.poll(0.1) == TwistCommand.stop()
