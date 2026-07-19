# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""The robot-side policy for pushing frames to the operator.

These tests pin the four decisions that make a live video push survive a bad
link, none of which need a socket to exercise.
"""

from __future__ import annotations

import pytest

from tritium_lib.fleet.frame_push import (
    FramePushPolicy,
    PushDecision,
    frame_push_path,
)


# ---------------------------------------------------------------------------
# the wire contract
# ---------------------------------------------------------------------------

def test_push_path_is_the_route_sc_serves():
    assert frame_push_path("isaac_rgb") == "/api/camera-feeds/sources/isaac_rgb/frame"


def test_push_path_refuses_an_empty_source_id():
    # A blank id would POST to .../sources//frame, which 404s at the far end
    # long after the rig has reported itself healthy.
    with pytest.raises(ValueError):
        frame_push_path("")


def test_push_path_escapes_a_slash_rather_than_forging_a_route():
    assert frame_push_path("a/b") == "/api/camera-feeds/sources/a%2Fb/frame"


# ---------------------------------------------------------------------------
# 1. decimation — never send faster than the operator asked for
# ---------------------------------------------------------------------------

def test_first_frame_is_always_offered():
    policy = FramePushPolicy(target_fps=10.0)
    assert policy.offer(now=0.0).send is True


def test_a_frame_inside_the_frame_interval_is_dropped():
    policy = FramePushPolicy(target_fps=10.0)  # 100 ms budget
    policy.offer(now=0.0)
    policy.sent(now=0.0)

    decision = policy.offer(now=0.05)
    assert decision.send is False
    assert decision.reason == "rate_limited"


def test_a_frame_past_the_frame_interval_is_sent():
    policy = FramePushPolicy(target_fps=10.0)
    policy.offer(now=0.0)
    policy.sent(now=0.0)

    assert policy.offer(now=0.10).send is True


# ---------------------------------------------------------------------------
# 2. drop, don't queue — staleness is worse than a gap for live video
# ---------------------------------------------------------------------------

def test_a_frame_offered_while_one_is_in_flight_is_dropped_not_queued():
    policy = FramePushPolicy(target_fps=1000.0)  # rate limit out of the way
    assert policy.offer(now=0.0).send is True  # this one is now in flight

    decision = policy.offer(now=1.0)
    assert decision.send is False
    assert decision.reason == "in_flight"


def test_completing_a_send_clears_the_in_flight_slot():
    policy = FramePushPolicy(target_fps=1000.0)
    policy.offer(now=0.0)
    policy.sent(now=0.0)

    assert policy.offer(now=1.0).send is True


def test_a_failed_send_also_clears_the_in_flight_slot():
    # Otherwise one dropped connection wedges the pusher forever.
    policy = FramePushPolicy(target_fps=1000.0, base_backoff_s=0.5)
    policy.offer(now=0.0)
    policy.failed(now=0.0)

    assert policy.offer(now=100.0).send is True


# ---------------------------------------------------------------------------
# 3. backoff — a downed operator must not be hammered every frame
# ---------------------------------------------------------------------------

def test_a_failure_backs_off_before_the_next_attempt():
    policy = FramePushPolicy(target_fps=1000.0, base_backoff_s=0.5)
    policy.offer(now=0.0)
    policy.failed(now=0.0)

    decision = policy.offer(now=0.1)
    assert decision.send is False
    assert decision.reason == "backoff"


def test_backoff_grows_exponentially_with_consecutive_failures():
    policy = FramePushPolicy(target_fps=1000.0, base_backoff_s=0.5)

    policy.offer(now=0.0)
    policy.failed(now=0.0)
    first = policy.backoff_remaining(now=0.0)

    policy.offer(now=10.0)  # past the first backoff
    policy.failed(now=10.0)
    second = policy.backoff_remaining(now=10.0)

    assert second > first
    assert second == pytest.approx(2 * first)


def test_backoff_is_capped_so_a_long_outage_still_recovers_promptly():
    policy = FramePushPolicy(target_fps=1000.0, base_backoff_s=0.5, max_backoff_s=4.0)
    now = 0.0
    for _ in range(20):
        policy.offer(now=now)
        policy.failed(now=now)
        now += 1000.0
    assert policy.backoff_remaining(now=now) <= 4.0


def test_one_success_clears_the_backoff():
    policy = FramePushPolicy(target_fps=1000.0, base_backoff_s=0.5)
    policy.offer(now=0.0)
    policy.failed(now=0.0)

    policy.offer(now=10.0)
    policy.sent(now=10.0)

    assert policy.backoff_remaining(now=10.0) == 0.0
    assert policy.stats.consecutive_failures == 0


# ---------------------------------------------------------------------------
# 4. honest stats — "pushing" must be distinguishable from "being refused"
# ---------------------------------------------------------------------------

def test_stats_separate_a_delivered_frame_from_a_dropped_one():
    policy = FramePushPolicy(target_fps=10.0)
    policy.offer(now=0.0)
    policy.sent(now=0.0)
    policy.offer(now=0.01)  # rate limited

    stats = policy.stats
    assert stats.sent == 1
    assert stats.dropped_rate_limited == 1
    assert stats.failed == 0


def test_a_rig_whose_every_push_is_refused_is_not_healthy():
    policy = FramePushPolicy(target_fps=1000.0, base_backoff_s=0.0)
    for i in range(5):
        policy.offer(now=float(i))
        policy.failed(now=float(i))

    assert policy.stats.sent == 0
    assert policy.stats.failed == 5
    assert policy.healthy is False


def test_a_rig_that_has_never_delivered_a_frame_is_not_healthy():
    # No failures either — it simply has not pushed anything yet.  "Nothing has
    # gone wrong" is not the same as "it is working", and a fresh pusher that
    # reports green is how a never-started thread stays invisible.
    policy = FramePushPolicy()
    assert policy.stats.failed == 0
    assert policy.healthy is False


def test_a_rig_that_is_delivering_is_healthy():
    policy = FramePushPolicy(target_fps=1000.0)
    policy.offer(now=0.0)
    policy.sent(now=0.0)
    assert policy.healthy is True


def test_sent_without_an_offer_is_a_programming_error_not_a_silent_stat():
    # Counting a frame nobody offered is exactly how a wedged pusher reports
    # throughput it never achieved.
    policy = FramePushPolicy()
    with pytest.raises(RuntimeError):
        policy.sent(now=0.0)


def test_target_fps_must_be_positive():
    with pytest.raises(ValueError):
        FramePushPolicy(target_fps=0.0)


def test_decision_is_immutable():
    decision = PushDecision(send=True, reason="ok")
    with pytest.raises(Exception):
        decision.send = False  # type: ignore[misc]
