# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Approximate-time pairing of two independently-arriving sensor streams.

The defect this exists to prevent, observed live against Isaac Sim on the
RTX 4090: a depth camera source pulled RGB and depth on two independent HTTP
threads and paired "latest of each" with no timestamps.  While the camera was
moving, the RGB frame the detector ran on and the depth frame the range was
read from were **different moments**, so the bbox landed on the wrong depth
pixels — a confident, wrong range on the tactical map.
"""

import pytest

from tritium_lib.perception import ApproximateTimeSync


class TestPairing:
    def test_pairs_nearest_in_time_not_newest(self):
        """The whole point: nearest-in-time, NOT latest-of-each."""
        s = ApproximateTimeSync(max_skew_s=0.05)
        s.push("rgb", "R@1.00", 1.00)
        s.push("depth", "D@0.99", 0.99)
        s.push("depth", "D@1.40", 1.40)  # newer, but far from the RGB frame

        pair = s.pair()
        assert pair is not None
        assert pair.values["rgb"] == "R@1.00"
        assert pair.values["depth"] == "D@0.99"   # NOT D@1.40
        assert pair.skew_s == pytest.approx(0.01, abs=1e-6)

    def test_rejects_a_pair_outside_the_skew_budget(self):
        s = ApproximateTimeSync(max_skew_s=0.05)
        s.push("rgb", "R", 1.00)
        s.push("depth", "D", 1.30)
        assert s.pair() is None

    def test_none_until_both_streams_have_arrived(self):
        s = ApproximateTimeSync(max_skew_s=0.05)
        assert s.pair() is None
        s.push("rgb", "R", 1.0)
        assert s.pair() is None

    def test_pairs_three_streams(self):
        """Stereo left/right + depth is the real rig, not just two channels."""
        s = ApproximateTimeSync(max_skew_s=0.05, keys=("left", "right", "depth"))
        s.push("left", "L", 1.00)
        s.push("right", "R", 1.02)
        s.push("depth", "D", 0.98)
        pair = s.pair()
        assert pair is not None
        assert set(pair.values) == {"left", "right", "depth"}
        assert pair.skew_s == pytest.approx(0.04, abs=1e-6)  # span 0.98..1.02

    def test_skew_is_the_full_span_not_a_single_delta(self):
        s = ApproximateTimeSync(max_skew_s=1.0, keys=("a", "b", "c"))
        s.push("a", 1, 0.0)
        s.push("b", 2, 0.3)
        s.push("c", 3, 0.9)
        assert s.pair().skew_s == pytest.approx(0.9, abs=1e-6)


class TestBuffering:
    def test_buffer_is_bounded(self):
        """A stalled consumer must not grow the buffer without limit."""
        s = ApproximateTimeSync(max_skew_s=0.05, depth=4)
        for i in range(50):
            s.push("rgb", i, float(i))
        assert len(s._buf["rgb"]) <= 4

    def test_old_frames_are_evicted_oldest_first(self):
        s = ApproximateTimeSync(max_skew_s=10.0, depth=3)
        for i in range(5):
            s.push("rgb", i, float(i))
        s.push("depth", "D", 4.0)
        # 0 and 1 evicted; the newest survivors remain
        assert s.pair().values["rgb"] == 4

    def test_pair_consumes_so_a_stalled_stream_cannot_re_emit_stale_frames(self):
        """The regression that motivated consuming semantics.

        If depth stalls, a non-destructive buffer would keep re-emitting the
        last good pair — stale data wearing a flattering skew number. After
        one emit the frames are gone, so a stalled stream yields ``None``.
        """
        s = ApproximateTimeSync(max_skew_s=0.05)
        s.push("rgb", "R", 1.0)
        s.push("depth", "D", 1.01)
        assert s.pair() is not None
        assert s.pair() is None                # consumed, not re-served
        s.push("rgb", "R2", 5.0)               # RGB alive, depth stalled
        assert s.pair() is None

    def test_serves_the_current_frame_and_discards_the_backlog(self):
        """A live feed wants *now*, not a replay of everything buffered.

        If a consumer falls behind, replaying the backlog in order would put
        it further behind on every tick and feed the tracker positions the
        robot has already driven past. The newest pair wins and the older
        frames are dropped with it.
        """
        s = ApproximateTimeSync(max_skew_s=0.05)
        s.push("rgb", "R1", 1.0)
        s.push("depth", "D1", 1.01)
        s.push("rgb", "R2", 2.0)
        s.push("depth", "D2", 2.01)
        assert s.pair().values["rgb"] == "R2"   # newest, not R1
        assert s.pair() is None                 # backlog dropped, not replayed

    def test_prefers_the_newest_qualifying_pair_over_the_tightest(self):
        """A slightly-looser fresh pair beats a tighter stale one."""
        s = ApproximateTimeSync(max_skew_s=0.05)
        s.push("rgb", "old", 1.000)
        s.push("depth", "old", 1.000)   # perfect 0.0 skew, but stale
        s.push("rgb", "new", 9.000)
        s.push("depth", "new", 9.030)   # 0.03 skew, but current
        pair = s.pair()
        assert pair.values["rgb"] == "new"


class TestGraceful:
    def test_unknown_key_is_ignored_not_raised(self):
        """One misconfigured channel must never kill a feed loop."""
        s = ApproximateTimeSync(max_skew_s=0.05)
        s.push("lidar", "X", 1.0)          # not a declared key
        s.push("rgb", "R", 1.0)
        s.push("depth", "D", 1.0)
        assert s.pair() is not None

    def test_non_finite_timestamp_is_dropped(self):
        s = ApproximateTimeSync(max_skew_s=0.05)
        s.push("rgb", "R", float("nan"))
        s.push("depth", "D", 1.0)
        assert s.pair() is None

    def test_none_payload_is_dropped(self):
        s = ApproximateTimeSync(max_skew_s=0.05)
        s.push("rgb", None, 1.0)
        s.push("depth", "D", 1.0)
        assert s.pair() is None


class TestStats:
    def test_tracks_paired_and_rejected_counts(self):
        s = ApproximateTimeSync(max_skew_s=0.05)
        s.push("rgb", "R", 1.0)
        s.push("depth", "D", 1.01)
        assert s.pair() is not None
        s.push("rgb", "R2", 5.0)
        s.push("depth", "D2", 9.0)
        assert s.pair() is None
        assert s.paired == 1
        assert s.rejected == 1

    def test_last_skew_is_reported_for_the_operator(self):
        s = ApproximateTimeSync(max_skew_s=0.05)
        s.push("rgb", "R", 1.0)
        s.push("depth", "D", 1.02)
        s.pair()
        assert s.last_skew_s == pytest.approx(0.02, abs=1e-6)
