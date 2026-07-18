"""Tests for path fidelity -- did the track the operator sees match reality?

The question this module answers is the one the whole Isaac->Tritium lane
turns on: a body walked somewhere, a track appeared on the tactical map, and
*nothing so far has compared the two*.  Every test below is written against a
way that comparison can lie.

Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC / AGPL-3.0
"""

import math

import pytest

from tritium_lib.geo.path_fidelity import (
    PathSample,
    compare_paths,
    path_length_m,
)


def _line(n=11, span_s=10.0, length_m=10.0, y=0.0, heading=0.0):
    """A straight walk along +X, evenly sampled."""
    return [
        PathSample(t=span_s * i / (n - 1), x=length_m * i / (n - 1), y=y,
                   heading_deg=heading)
        for i in range(n)
    ]


class TestPathLength:
    def test_straight_line_length_is_the_span(self):
        assert path_length_m(_line()) == pytest.approx(10.0)

    def test_stationary_path_has_zero_length(self):
        samples = [PathSample(t=float(i), x=3.0, y=4.0) for i in range(5)]
        assert path_length_m(samples) == 0.0

    def test_single_sample_has_zero_length(self):
        assert path_length_m([PathSample(t=0.0, x=1.0, y=1.0)]) == 0.0

    def test_length_is_arc_not_displacement(self):
        # Out 5 m and back: displacement 0, arc length 10.
        out_and_back = [
            PathSample(t=0.0, x=0.0, y=0.0),
            PathSample(t=1.0, x=5.0, y=0.0),
            PathSample(t=2.0, x=0.0, y=0.0),
        ]
        assert path_length_m(out_and_back) == pytest.approx(10.0)


class TestPerfectAgreement:
    def test_identical_paths_have_zero_error(self):
        ref = _line()
        result = compare_paths(ref, list(ref))
        assert result.max_error_m == pytest.approx(0.0)
        assert result.rms_error_m == pytest.approx(0.0)
        assert result.verdict == "AGREES"

    def test_offset_sample_times_are_interpolated_not_rejected(self):
        # The reported track samples BETWEEN the reference samples.  A naive
        # index-to-index comparison would either crash or silently compare
        # mismatched pairs; interpolation is what makes this exact.
        ref = _line(n=11)
        reported = [PathSample(t=0.5 + i, x=0.5 + i, y=0.0) for i in range(9)]
        result = compare_paths(ref, reported)
        assert result.samples_compared == 9
        assert result.max_error_m == pytest.approx(0.0, abs=1e-9)

    def test_different_sample_rates_still_align(self):
        ref = _line(n=101)
        reported = _line(n=6)
        result = compare_paths(ref, reported)
        assert result.max_error_m == pytest.approx(0.0, abs=1e-9)
        assert result.verdict == "AGREES"


class TestWaysAComparisonCanLie:
    def test_stationary_report_of_a_moving_body_is_not_agreement(self):
        # The classic silent failure: ingest wedges, the map keeps painting
        # the first fix, and every "is there a track?" check still passes.
        ref = _line(length_m=10.0)
        reported = [PathSample(t=s.t, x=0.0, y=0.0) for s in ref]
        result = compare_paths(ref, reported)
        assert result.reported_length_m == pytest.approx(0.0)
        assert result.length_ratio == pytest.approx(0.0)
        assert result.verdict == "DIVERGED"

    def test_a_short_walk_does_not_let_a_frozen_track_hide_under_tolerance(self):
        # Found by the live negative control, not by inspection.  A tumbling
        # Go2 covered 2.1 m of PATH but stayed within 0.39 m of where it
        # started, so a frozen ingest scored max_error 0.39 m -- inside a
        # 0.50 m tolerance -- and came back AGREES while the reported path
        # was 16% of the truth.  Displacement is what bounds the error here;
        # arc length is what exposes the lie, which is exactly why the two
        # are measured separately.
        # One 0.22 m-radius loop back to the start: 1.38 m of path, never
        # more than 0.44 m from the origin.
        ref = [
            PathSample(t=0.4 * i,
                       x=0.22 * math.sin(i * 2 * math.pi / 20),
                       y=0.22 * math.cos(i * 2 * math.pi / 20) - 0.22)
            for i in range(21)
        ]
        reported = [PathSample(t=s.t, x=0.0, y=0.0) for s in ref]
        result = compare_paths(ref, reported, tolerance_m=0.50)
        assert result.max_error_m < 0.50  # the trap: error alone says fine
        assert result.length_ratio < 0.5
        assert result.verdict == "DIVERGED"

    def test_a_grossly_short_track_is_not_agreement(self):
        # Reported track follows the truth but covers a third of the ground
        # -- a decimated or stalling feed.  Small absolute error, wrong shape.
        ref = _line(n=21, span_s=10.0, length_m=1.5)
        reported = [PathSample(t=s.t, x=min(s.x, 0.5), y=s.y) for s in ref]
        result = compare_paths(ref, reported, tolerance_m=0.50)
        assert result.length_ratio < 0.5
        assert result.verdict == "DIVERGED"

    def test_shape_guard_does_not_fire_on_a_body_that_barely_moved(self):
        # If the reference itself covered almost no ground there is no shape
        # to check, and a frozen report is legitimately correct.  The guard
        # must not manufacture a failure here.
        ref = _line(n=11, span_s=5.0, length_m=0.20)
        reported = [PathSample(t=s.t, x=0.0, y=0.0) for s in ref]
        result = compare_paths(ref, reported, tolerance_m=0.50)
        assert result.verdict == "AGREES"

    def test_time_shifted_replay_of_the_truth_is_caught(self):
        # Reporting the reference verbatim but 2 s late has perfect SHAPE --
        # length ratio is 1.0 and every reported point lies exactly on the
        # reference path.  Only time-aligned comparison sees the lag.
        ref = _line(n=21, span_s=10.0, length_m=10.0)
        reported = [PathSample(t=s.t + 2.0, x=s.x, y=s.y) for s in ref]
        result = compare_paths(ref, reported)
        assert result.length_ratio == pytest.approx(1.0, abs=1e-6)
        assert result.max_error_m > 1.5  # 2 s at 1 m/s
        assert result.verdict != "AGREES"

    def test_mirrored_path_has_matching_length_but_diverges(self):
        # Same distance travelled, wrong direction -- a sign-flip bug in a
        # frame conversion looks exactly like this.
        ref = _line(length_m=10.0)
        reported = [PathSample(t=s.t, x=-s.x, y=s.y) for s in ref]
        result = compare_paths(ref, reported)
        assert result.reported_length_m == pytest.approx(result.reference_length_m)
        assert result.verdict == "DIVERGED"

    def test_axis_swap_is_caught(self):
        # x/y transposed: a real and very quiet ENU-vs-NED style mistake.
        ref = [PathSample(t=float(i), x=float(i), y=0.5 * i) for i in range(11)]
        reported = [PathSample(t=s.t, x=s.y, y=s.x) for s in ref]
        result = compare_paths(ref, reported)
        assert result.max_error_m > 1.0
        assert result.verdict != "AGREES"

    def test_a_single_lucky_sample_does_not_carry_the_verdict(self):
        # Reported track is right at t=0 and wrong everywhere after.  A
        # mean-only metric can be dragged toward passing; max_error cannot.
        ref = _line(n=11)
        reported = [PathSample(t=s.t, x=s.x if s.t == 0.0 else 0.0, y=0.0)
                    for s in ref]
        result = compare_paths(ref, reported)
        assert result.max_error_m == pytest.approx(10.0)
        assert result.verdict == "DIVERGED"


class TestTolerance:
    def test_small_jitter_within_tolerance_still_agrees(self):
        ref = _line(n=11)
        reported = [PathSample(t=s.t, x=s.x + 0.02, y=s.y - 0.01) for s in ref]
        result = compare_paths(ref, reported, tolerance_m=0.10)
        assert result.verdict == "AGREES"

    def test_drift_beyond_tolerance_but_within_gross_error_is_DRIFTS(self):
        ref = _line(n=11)
        reported = [PathSample(t=s.t, x=s.x + 0.30, y=s.y) for s in ref]
        result = compare_paths(ref, reported, tolerance_m=0.10, diverged_m=2.0)
        assert result.verdict == "DRIFTS"

    def test_tolerance_is_applied_to_max_not_mean(self):
        # One 5 m excursion in an otherwise perfect track must not pass just
        # because the average stayed small.
        ref = _line(n=21)
        reported = [PathSample(t=s.t, x=s.x + (5.0 if i == 10 else 0.0), y=s.y)
                    for i, s in enumerate(ref)]
        result = compare_paths(ref, reported, tolerance_m=0.10)
        assert result.mean_error_m < 0.5
        assert result.verdict == "DIVERGED"


class TestHeading:
    def test_matching_headings_report_zero_error(self):
        ref = _line(heading=90.0)
        reported = [PathSample(t=s.t, x=s.x, y=s.y, heading_deg=90.0) for s in ref]
        result = compare_paths(ref, reported)
        assert result.max_heading_error_deg == pytest.approx(0.0)

    def test_heading_error_wraps_the_short_way(self):
        # 359 deg vs 1 deg is a 2 deg error, not 358.
        ref = [PathSample(t=0.0, x=0.0, y=0.0, heading_deg=359.0)]
        reported = [PathSample(t=0.0, x=0.0, y=0.0, heading_deg=1.0)]
        result = compare_paths(ref, reported)
        assert result.max_heading_error_deg == pytest.approx(2.0)

    def test_backwards_heading_is_a_180_error(self):
        ref = [PathSample(t=0.0, x=0.0, y=0.0, heading_deg=45.0)]
        reported = [PathSample(t=0.0, x=0.0, y=0.0, heading_deg=225.0)]
        result = compare_paths(ref, reported)
        assert result.max_heading_error_deg == pytest.approx(180.0)

    def test_missing_headings_leave_the_metric_none_not_zero(self):
        # Absent data must never read as perfect agreement.
        ref = _line(heading=None)
        reported = [PathSample(t=s.t, x=s.x, y=s.y) for s in ref]
        result = compare_paths(ref, reported)
        assert result.max_heading_error_deg is None
        assert result.heading_samples_compared == 0

    def test_partial_headings_score_only_the_pairs_that_have_both(self):
        ref = [PathSample(t=float(i), x=float(i), y=0.0,
                          heading_deg=0.0 if i < 3 else None)
               for i in range(6)]
        reported = [PathSample(t=float(i), x=float(i), y=0.0, heading_deg=10.0)
                    for i in range(6)]
        result = compare_paths(ref, reported)
        assert result.heading_samples_compared == 3
        assert result.max_heading_error_deg == pytest.approx(10.0)


class TestOverlapAndDegenerateInput:
    def test_no_time_overlap_is_its_own_verdict(self):
        ref = _line(span_s=10.0)
        reported = [PathSample(t=100.0 + i, x=float(i), y=0.0) for i in range(5)]
        result = compare_paths(ref, reported)
        assert result.verdict == "NO_OVERLAP"
        assert result.samples_compared == 0

    def test_reported_samples_outside_the_reference_span_are_dropped(self):
        # Extrapolating past the end of the truth invents error that isn't
        # measured, so those samples are excluded rather than clamped.
        ref = _line(n=11, span_s=10.0)
        reported = [PathSample(t=t, x=t, y=0.0)
                    for t in (-5.0, 0.0, 5.0, 10.0, 25.0)]
        result = compare_paths(ref, reported)
        assert result.samples_compared == 3

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError):
            compare_paths([], _line())

    def test_empty_report_is_NO_OVERLAP_not_a_crash(self):
        # A body that walked and produced no track at all is a real outcome
        # of this pipeline, and it must be reportable.
        result = compare_paths(_line(), [])
        assert result.verdict == "NO_OVERLAP"
        assert result.samples_compared == 0

    def test_unsorted_input_is_sorted_not_trusted(self):
        ref = _line(n=11)
        reported = list(reversed([PathSample(t=s.t, x=s.x, y=s.y) for s in ref]))
        result = compare_paths(ref, reported)
        assert result.max_error_m == pytest.approx(0.0, abs=1e-9)

    def test_zero_length_reference_gives_no_length_ratio(self):
        # Dividing by a stationary reference would be a ZeroDivisionError or,
        # worse, an inf that reads as a huge disagreement.
        ref = [PathSample(t=float(i), x=0.0, y=0.0) for i in range(5)]
        reported = [PathSample(t=float(i), x=0.0, y=0.0) for i in range(5)]
        result = compare_paths(ref, reported)
        assert result.length_ratio is None
        assert result.verdict == "AGREES"


class TestReportShape:
    def test_summary_names_the_numbers_a_reader_needs(self):
        result = compare_paths(_line(), _line())
        text = result.summary()
        assert "AGREES" in text
        for token in ("max", "rms", "samples"):
            assert token in text.lower()

    def test_as_dict_round_trips_through_json(self):
        import json

        result = compare_paths(_line(), _line())
        assert json.loads(json.dumps(result.as_dict()))["verdict"] == "AGREES"

    def test_final_error_is_reported_separately_from_max(self):
        # A track that drifts and then snaps back at the end is a different
        # failure from one that ends far away; both need to be visible.
        ref = _line(n=11)
        reported = [PathSample(t=s.t, x=s.x + (3.0 if 2 <= i <= 5 else 0.0), y=s.y)
                    for i, s in enumerate(ref)]
        result = compare_paths(ref, reported)
        assert result.max_error_m == pytest.approx(3.0)
        assert result.final_error_m == pytest.approx(0.0)


class TestNoHeavyDependencies:
    def test_module_imports_without_numpy(self):
        # This must import on a bare Jetson and inside a simulator's embedded
        # Python alike, so it is stdlib-only by contract.
        import tritium_lib.geo.path_fidelity as mod

        source = open(mod.__file__).read()
        assert "import numpy" not in source
        assert "math" in source


class TestRealisticShape:
    def test_a_noisy_but_honest_track_agrees(self):
        # What a healthy ingest actually looks like: right shape, small noise.
        ref = _line(n=41, span_s=8.0, length_m=8.0)
        reported = [
            PathSample(t=s.t, x=s.x + 0.03 * math.sin(i), y=s.y + 0.02 * math.cos(i))
            for i, s in enumerate(ref)
        ]
        result = compare_paths(ref, reported, tolerance_m=0.10)
        assert result.verdict == "AGREES"
        assert result.length_ratio == pytest.approx(1.0, abs=0.5)
