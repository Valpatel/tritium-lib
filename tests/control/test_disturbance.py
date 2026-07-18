# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for deterministic disturbance injection and recovery scoring.

The point of this module is to make a controller's disturbance rejection
*measurable*, so the tests are mostly about the ways a disturbance experiment
can silently become vacuous: a kick that never fired, a kick that fired twice,
a body scored as recovered because the trace ended while it was still falling.
"""

from __future__ import annotations

import pytest

from tritium_lib.control.disturbance import (
    DisturbanceSchedule,
    Impulse,
    RecoveryScore,
    score_recovery,
)


# --------------------------------------------------------------------------
# Impulse
# --------------------------------------------------------------------------

def test_impulse_defaults_to_no_push():
    imp = Impulse(at_time=1.0)
    assert imp.linear == (0.0, 0.0, 0.0)
    assert imp.angular == (0.0, 0.0, 0.0)


def test_impulse_is_frozen_so_a_schedule_cannot_be_mutated_mid_run():
    imp = Impulse(at_time=1.0, linear=(10.0, 0.0, 0.0))
    with pytest.raises(Exception):
        imp.at_time = 2.0  # type: ignore[misc]


def test_impulse_rejects_negative_time():
    with pytest.raises(ValueError):
        Impulse(at_time=-0.1)


def test_impulse_magnitude_is_the_euclidean_norm_of_the_linear_part():
    assert Impulse(at_time=1.0, linear=(3.0, 4.0, 0.0)).magnitude == pytest.approx(5.0)


def test_a_zero_impulse_is_not_a_disturbance():
    # An experiment configured with a zero kick is an experiment that measures
    # nothing; callers need to be able to detect that without inspecting axes.
    assert Impulse(at_time=1.0).is_zero is True
    assert Impulse(at_time=1.0, angular=(0.0, 0.5, 0.0)).is_zero is False


# --------------------------------------------------------------------------
# DisturbanceSchedule — firing discipline
# --------------------------------------------------------------------------

def test_impulse_fires_in_the_window_that_contains_its_time():
    sched = DisturbanceSchedule([Impulse(at_time=2.0, linear=(5.0, 0.0, 0.0))])
    assert sched.due(0.0, 1.9) == []
    fired = sched.due(1.9, 2.1)
    assert [f.at_time for f in fired] == [2.0]


def test_impulse_fires_exactly_once_across_many_windows():
    sched = DisturbanceSchedule([Impulse(at_time=2.0, linear=(5.0, 0.0, 0.0))])
    sched.due(1.9, 2.1)
    assert sched.due(2.1, 2.3) == []
    assert sched.due(2.3, 9.0) == []


def test_a_step_that_jumps_clean_over_the_time_still_fires():
    # A slow frame must not silently cancel the disturbance.  If it did, the
    # A/B would compare two undisturbed runs and report a null result that
    # looks like successful rejection.
    sched = DisturbanceSchedule([Impulse(at_time=2.0, linear=(5.0, 0.0, 0.0))])
    fired = sched.due(0.0, 8.0)
    assert len(fired) == 1


def test_a_window_that_STARTS_past_the_time_still_fires():
    # The sharpest version of the missed-kick bug, and the one a "jumped over"
    # test starting at t=0 does not reach: the driver's first sample lands
    # after the scheduled time (a slow scene build, a late first callback), so
    # every window the schedule ever sees already begins in the past.  Firing
    # must depend on the window's END, never on its start.
    sched = DisturbanceSchedule([Impulse(at_time=2.0, linear=(5.0, 0.0, 0.0))])
    fired = sched.due(3.5, 3.6)
    assert len(fired) == 1, "a late first window silently cancelled the kick"


def test_a_late_start_fires_every_overdue_impulse_not_just_one():
    sched = DisturbanceSchedule([
        Impulse(at_time=1.0, linear=(1.0, 0.0, 0.0)),
        Impulse(at_time=2.0, linear=(2.0, 0.0, 0.0)),
    ])
    assert len(sched.due(5.0, 5.1)) == 2


def test_multiple_impulses_fire_in_time_order_within_one_window():
    sched = DisturbanceSchedule([
        Impulse(at_time=3.0, linear=(1.0, 0.0, 0.0)),
        Impulse(at_time=1.0, linear=(2.0, 0.0, 0.0)),
    ])
    fired = sched.due(0.0, 5.0)
    assert [f.at_time for f in fired] == [1.0, 3.0]


def test_pending_and_fired_expose_whether_the_experiment_actually_happened():
    sched = DisturbanceSchedule([Impulse(at_time=2.0, linear=(5.0, 0.0, 0.0))])
    assert sched.pending == 1
    assert sched.fired == ()
    sched.due(1.9, 2.1)
    assert sched.pending == 0
    assert len(sched.fired) == 1


def test_a_run_that_ends_before_the_kick_reports_it_as_unfired():
    # The single most dangerous outcome: the run is short, the kick never
    # lands, and both arms score perfectly.  This must be loud.
    sched = DisturbanceSchedule([Impulse(at_time=9.0, linear=(5.0, 0.0, 0.0))])
    sched.due(0.0, 6.0)
    assert sched.pending == 1
    assert sched.all_fired is False


def test_reset_rearms_the_schedule_for_the_next_trial():
    # Trials rebuild the scene; a schedule that stayed spent would disturb
    # only trial 1 and leave the rest of the sample undisturbed.
    sched = DisturbanceSchedule([Impulse(at_time=2.0, linear=(5.0, 0.0, 0.0))])
    sched.due(0.0, 3.0)
    sched.reset()
    assert sched.pending == 1
    assert len(sched.due(0.0, 3.0)) == 1


def test_schedule_rejects_an_empty_impulse_list():
    with pytest.raises(ValueError):
        DisturbanceSchedule([])


def test_backwards_time_is_rejected_rather_than_silently_refiring():
    sched = DisturbanceSchedule([Impulse(at_time=2.0, linear=(5.0, 0.0, 0.0))])
    with pytest.raises(ValueError):
        sched.due(3.0, 1.0)


# --------------------------------------------------------------------------
# score_recovery
# --------------------------------------------------------------------------

def _trace(pairs):
    return [(float(t), float(v)) for t, v in pairs]


def test_peak_is_measured_after_the_disturbance_not_over_the_whole_run():
    # A big pre-kick excursion belongs to the gait, not to the disturbance.
    samples = _trace([(0.0, 30.0), (1.0, 2.0), (2.0, 9.0), (3.0, 1.0)])
    score = score_recovery(samples, disturbed_at=1.5, settled_below_deg=5.0,
                           hold_for=1.0)
    assert score.peak_deg == pytest.approx(9.0)
    assert score.peak_at == pytest.approx(2.0)
    assert score.baseline_deg == pytest.approx(30.0)


def test_a_body_that_returns_below_threshold_and_stays_there_is_recovered():
    samples = _trace([(0.0, 1.0), (1.0, 20.0), (2.0, 8.0),
                      (3.0, 3.0), (4.0, 2.0), (5.0, 2.0)])
    score = score_recovery(samples, disturbed_at=0.5, settled_below_deg=5.0,
                           hold_for=1.5)
    assert score.recovered is True
    # Settling is timed from the disturbance, not from the trace start.
    assert score.settle_time == pytest.approx(2.5)


def test_touching_the_threshold_briefly_is_not_settling():
    # Oscillating through the band is not recovery.  Settling must be the
    # first time it goes below AND STAYS below -- the same "worst sample, not
    # mean" discipline as path_fidelity.
    samples = _trace([(0.0, 1.0), (1.0, 25.0), (2.0, 3.0), (3.0, 22.0),
                      (4.0, 2.0), (5.0, 1.0), (6.0, 1.0)])
    score = score_recovery(samples, disturbed_at=0.5, settled_below_deg=5.0,
                           hold_for=1.5)
    assert score.recovered is True
    assert score.settle_time == pytest.approx(3.5)  # from t=4.0, not t=2.0


def test_a_trace_that_ends_while_still_tilted_did_not_recover():
    samples = _trace([(0.0, 1.0), (1.0, 30.0), (2.0, 45.0), (3.0, 80.0)])
    score = score_recovery(samples, disturbed_at=0.5, settled_below_deg=5.0,
                           hold_for=1.0)
    assert score.recovered is False
    assert score.settle_time is None


def test_settle_time_is_none_rather_than_a_large_number_when_never_settled():
    # A sentinel like 999.0 would average into a summary and read as a slow
    # recovery instead of no recovery at all.
    samples = _trace([(0.0, 1.0), (1.0, 60.0), (2.0, 60.0)])
    score = score_recovery(samples, disturbed_at=0.5, settled_below_deg=5.0,
                           hold_for=0.5)
    assert score.settle_time is None


def test_a_trace_too_short_to_prove_the_hold_is_not_called_recovered():
    # Below threshold at the final sample proves nothing if the trace stops
    # immediately after; insufficient evidence is not success.
    samples = _trace([(0.0, 1.0), (1.0, 25.0), (2.0, 2.0)])
    score = score_recovery(samples, disturbed_at=0.5, settled_below_deg=5.0,
                           hold_for=3.0)
    assert score.recovered is False


def test_no_samples_after_the_disturbance_raises():
    # Scoring this would report peak 0 and read as flawless rejection.
    samples = _trace([(0.0, 1.0), (1.0, 2.0)])
    with pytest.raises(ValueError):
        score_recovery(samples, disturbed_at=5.0, settled_below_deg=5.0,
                       hold_for=1.0)


def test_empty_trace_raises():
    with pytest.raises(ValueError):
        score_recovery([], disturbed_at=1.0, settled_below_deg=5.0,
                       hold_for=1.0)


def test_unsorted_samples_are_rejected_rather_than_silently_scored():
    samples = _trace([(0.0, 1.0), (2.0, 5.0), (1.0, 40.0)])
    with pytest.raises(ValueError):
        score_recovery(samples, disturbed_at=0.5, settled_below_deg=5.0,
                       hold_for=1.0)


def test_negative_threshold_is_rejected():
    samples = _trace([(0.0, 1.0), (1.0, 5.0)])
    with pytest.raises(ValueError):
        score_recovery(samples, disturbed_at=0.5, settled_below_deg=-1.0,
                       hold_for=1.0)


def test_baseline_is_zero_when_the_disturbance_precedes_every_sample():
    samples = _trace([(1.0, 10.0), (2.0, 3.0), (3.0, 2.0)])
    score = score_recovery(samples, disturbed_at=0.0, settled_below_deg=5.0,
                           hold_for=1.0)
    assert score.baseline_deg == pytest.approx(0.0)


def test_excursion_reports_how_much_worse_the_kick_made_it():
    samples = _trace([(0.0, 4.0), (1.0, 19.0), (2.0, 2.0), (3.0, 2.0)])
    score = score_recovery(samples, disturbed_at=0.5, settled_below_deg=5.0,
                           hold_for=1.0)
    assert score.excursion_deg == pytest.approx(15.0)


def test_score_is_json_friendly_for_the_run_report():
    samples = _trace([(0.0, 1.0), (1.0, 20.0), (2.0, 2.0), (3.0, 2.0)])
    score = score_recovery(samples, disturbed_at=0.5, settled_below_deg=5.0,
                           hold_for=1.0)
    payload = score.as_dict()
    assert payload["recovered"] is True
    assert payload["peak_deg"] == pytest.approx(20.0)
    assert isinstance(score, RecoveryScore)


def test_samples_after_counts_only_post_disturbance_evidence():
    samples = _trace([(0.0, 1.0), (1.0, 2.0), (2.0, 20.0), (3.0, 2.0)])
    score = score_recovery(samples, disturbed_at=1.5, settled_below_deg=5.0,
                           hold_for=1.0)
    assert score.samples_after == 2
