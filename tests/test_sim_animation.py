# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine animation system — easing, keyframes, tracks, library, interpolation buffer."""

import math
import time

import pytest

from tritium_lib.sim_engine.animation import (
    AnimationLibrary,
    AnimationTrack,
    EntityAnimation,
    InterpolationBuffer,
    Keyframe,
    back,
    bounce,
    ease_in,
    ease_in_out,
    ease_out,
    elastic,
    get_easing,
    lerp,
    linear,
)


# ── Easing functions ────────────────────────────────────────────────

class TestEasingFunctions:
    """Tests for easing functions."""

    @pytest.mark.parametrize("fn", [linear, ease_in, ease_out, ease_in_out, bounce, elastic, back])
    def test_zero_returns_zero_or_near(self, fn):
        result = fn(0.0)
        assert result == pytest.approx(0.0, abs=0.01)

    @pytest.mark.parametrize("fn", [linear, ease_in, ease_out, ease_in_out, bounce, elastic])
    def test_one_returns_one_or_near(self, fn):
        result = fn(1.0)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_linear_midpoint(self):
        assert linear(0.5) == pytest.approx(0.5)

    def test_ease_in_slower_start(self):
        assert ease_in(0.5) < 0.5

    def test_ease_out_faster_start(self):
        assert ease_out(0.5) > 0.5

    def test_ease_in_out_midpoint(self):
        assert ease_in_out(0.5) == pytest.approx(0.5, abs=0.05)

    def test_bounce_bounces(self):
        v1 = bounce(0.8)
        v2 = bounce(0.9)
        # Bounce should have non-monotonic behavior near end
        assert v1 > 0.5

    def test_elastic_overshoots(self):
        # Elastic should go past 1.0 at some point
        values = [elastic(t / 20.0) for t in range(21)]
        assert max(values) > 1.0

    def test_back_undershoots(self):
        # Back easing goes below 0 initially
        v = back(0.2)
        assert v < 0.0

    @pytest.mark.parametrize("fn", [linear, ease_in, ease_out, ease_in_out, bounce, elastic, back])
    def test_clamping_below_zero(self, fn):
        result = fn(-0.5)
        # All functions clamp input to [0,1] first
        assert result == pytest.approx(fn(0.0), abs=0.01)

    @pytest.mark.parametrize("fn", [linear, ease_in, ease_out, ease_in_out, bounce, elastic, back])
    def test_clamping_above_one(self, fn):
        result = fn(1.5)
        assert result == pytest.approx(fn(1.0), abs=0.01)


class TestGetEasing:
    def test_known_names(self):
        assert get_easing("linear") is linear
        assert get_easing("ease_in") is ease_in
        assert get_easing("bounce") is bounce

    def test_unknown_defaults_to_linear(self):
        assert get_easing("nonexistent") is linear


# ── Keyframe ────────────────────────────────────────────────────────

class TestKeyframe:
    def test_scalar_to_dict(self):
        kf = Keyframe(time=0.5, value=42.0, easing="ease_out")
        d = kf.to_dict()
        assert d["time"] == 0.5
        assert d["value"] == 42.0
        assert d["easing"] == "ease_out"

    def test_vec3_to_dict(self):
        kf = Keyframe(time=1.0, value=(1.0, 2.0, 3.0))
        d = kf.to_dict()
        assert d["value"] == [1.0, 2.0, 3.0]

    def test_default_easing(self):
        kf = Keyframe(time=0.0, value=0.0)
        assert kf.easing == "linear"


# ── AnimationTrack ──────────────────────────────────────────────────

class TestAnimationTrack:
    def test_empty_track(self):
        track = AnimationTrack("empty")
        assert track.duration == 0.0
        assert track.evaluate(0.5) == 0.0

    def test_single_keyframe(self):
        track = AnimationTrack("single", [Keyframe(0.0, 5.0)])
        assert track.evaluate(0.0) == 5.0
        assert track.evaluate(1.0) == 5.0  # Clamp after last

    def test_linear_interpolation(self):
        track = AnimationTrack("pos", [
            Keyframe(0.0, 0.0),
            Keyframe(1.0, 10.0),
        ])
        assert track.evaluate(0.0) == pytest.approx(0.0)
        assert track.evaluate(0.5) == pytest.approx(5.0)
        assert track.evaluate(1.0) == pytest.approx(10.0)

    def test_before_first_keyframe(self):
        track = AnimationTrack("pos", [
            Keyframe(1.0, 5.0),
            Keyframe(2.0, 10.0),
        ])
        assert track.evaluate(0.0) == 5.0

    def test_after_last_keyframe(self):
        track = AnimationTrack("pos", [
            Keyframe(0.0, 0.0),
            Keyframe(1.0, 10.0),
        ])
        assert track.evaluate(5.0) == 10.0

    def test_vec2_interpolation(self):
        track = AnimationTrack("pos", [
            Keyframe(0.0, (0.0, 0.0)),
            Keyframe(1.0, (10.0, 20.0)),
        ])
        result = track.evaluate(0.5)
        assert len(result) == 2
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(10.0)

    def test_vec3_interpolation(self):
        track = AnimationTrack("pos", [
            Keyframe(0.0, (0.0, 0.0, 0.0)),
            Keyframe(1.0, (10.0, 20.0, 30.0)),
        ])
        result = track.evaluate(0.5)
        assert len(result) == 3
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(10.0)
        assert result[2] == pytest.approx(15.0)

    def test_duration(self):
        track = AnimationTrack("pos", [
            Keyframe(1.0, 0.0),
            Keyframe(3.0, 10.0),
        ])
        assert track.duration == 2.0

    def test_start_time(self):
        track = AnimationTrack("pos", [
            Keyframe(2.0, 0.0),
            Keyframe(5.0, 10.0),
        ])
        assert track.start_time == 2.0

    def test_end_time(self):
        track = AnimationTrack("pos", [
            Keyframe(2.0, 0.0),
            Keyframe(5.0, 10.0),
        ])
        assert track.end_time == 5.0

    def test_keyframes_sorted(self):
        track = AnimationTrack("pos")
        track.add_keyframe(Keyframe(2.0, 20.0))
        track.add_keyframe(Keyframe(0.0, 0.0))
        track.add_keyframe(Keyframe(1.0, 10.0))
        kfs = track.keyframes
        assert kfs[0].time == 0.0
        assert kfs[1].time == 1.0
        assert kfs[2].time == 2.0

    def test_to_dict(self):
        track = AnimationTrack("test", [
            Keyframe(0.0, 0.0),
            Keyframe(1.0, 10.0),
        ])
        d = track.to_dict()
        assert d["name"] == "test"
        assert d["duration"] == 1.0
        assert len(d["keyframes"]) == 2


# ── EntityAnimation ─────────────────────────────────────────────────

class TestEntityAnimation:
    def test_empty_animation(self):
        anim = EntityAnimation("empty")
        assert anim.duration == 0.0
        result = anim.evaluate(0.5)
        assert result == {}

    def test_multi_track(self):
        anim = EntityAnimation("test")
        anim.add_track("x", AnimationTrack("x", [
            Keyframe(0.0, 0.0), Keyframe(1.0, 10.0),
        ]))
        anim.add_track("y", AnimationTrack("y", [
            Keyframe(0.0, 0.0), Keyframe(1.0, 20.0),
        ]))
        result = anim.evaluate(0.5)
        assert result["x"] == pytest.approx(5.0)
        assert result["y"] == pytest.approx(10.0)

    def test_looping(self):
        anim = EntityAnimation("loop", loop=True)
        anim.add_track("x", AnimationTrack("x", [
            Keyframe(0.0, 0.0), Keyframe(1.0, 10.0),
        ]))
        # At t=1.5, loop means t=0.5
        result = anim.evaluate(1.5)
        assert result["x"] == pytest.approx(5.0)

    def test_to_three_js(self):
        anim = EntityAnimation("test", loop=True)
        anim.add_track("pos", AnimationTrack("pos", [
            Keyframe(0.0, 0.0), Keyframe(2.0, 10.0),
        ]))
        d = anim.to_three_js()
        assert d["name"] == "test"
        assert d["duration"] == 2.0
        assert d["loop"] is True
        assert "pos" in d["tracks"]


# ── AnimationLibrary ────────────────────────────────────────────────

class TestAnimationLibrary:
    def test_list_animations(self):
        names = AnimationLibrary.list_animations()
        assert "walk_cycle" in names
        assert "death_fall" in names
        assert "explosion_shake" in names
        assert len(names) >= 10

    def test_get_known_animation(self):
        anim = AnimationLibrary.get("walk_cycle")
        assert anim is not None
        assert anim.name == "walk_cycle"
        assert anim.loop is True
        assert anim.duration > 0

    def test_get_unknown_returns_none(self):
        assert AnimationLibrary.get("nonexistent") is None

    def test_death_fall_not_looping(self):
        anim = AnimationLibrary.get("death_fall")
        assert anim.loop is False

    def test_fire_flicker_looping(self):
        anim = AnimationLibrary.get("fire_flicker")
        assert anim.loop is True

    def test_smoke_rise_not_looping(self):
        anim = AnimationLibrary.get("smoke_rise")
        assert anim.loop is False

    def test_custom_duration(self):
        anim = AnimationLibrary.get("walk_cycle", duration=2.0)
        assert anim.duration == pytest.approx(2.0)

    @pytest.mark.parametrize("name", AnimationLibrary.list_animations())
    def test_all_animations_evaluable(self, name):
        anim = AnimationLibrary.get(name)
        assert anim is not None
        result = anim.evaluate(anim.duration / 2)
        assert isinstance(result, dict)


# ── Lerp ────────────────────────────────────────────────────────────

class TestLerp:
    def test_t_zero(self):
        assert lerp(0.0, 10.0, 0.0) == 0.0

    def test_t_one(self):
        assert lerp(0.0, 10.0, 1.0) == 10.0

    def test_midpoint(self):
        assert lerp(0.0, 10.0, 0.5) == 5.0

    def test_negative_values(self):
        assert lerp(-10.0, 10.0, 0.5) == 0.0


# ── InterpolationBuffer ────────────────────────────────────────────

class TestInterpolationBuffer:
    def test_empty_returns_none(self):
        buf = InterpolationBuffer()
        assert buf.evaluate(now=1.0) is None
        assert buf.empty is True

    def test_single_sample_returned(self):
        buf = InterpolationBuffer()
        buf.push(5.0, timestamp=1.0)
        assert buf.evaluate(now=1.5) == 5.0

    def test_interpolation_between_samples(self):
        buf = InterpolationBuffer(delay=0.0)
        buf.push(0.0, timestamp=0.0)
        buf.push(10.0, timestamp=1.0)
        result = buf.evaluate(now=0.5)
        assert result == pytest.approx(5.0)

    def test_before_all_samples(self):
        buf = InterpolationBuffer(delay=0.0)
        buf.push(5.0, timestamp=1.0)
        buf.push(10.0, timestamp=2.0)
        assert buf.evaluate(now=0.0) == 5.0

    def test_after_all_samples(self):
        buf = InterpolationBuffer(delay=0.0)
        buf.push(5.0, timestamp=1.0)
        buf.push(10.0, timestamp=2.0)
        assert buf.evaluate(now=5.0) == 10.0

    def test_delay_shifts_render_time(self):
        buf = InterpolationBuffer(delay=0.5)
        buf.push(0.0, timestamp=0.0)
        buf.push(10.0, timestamp=1.0)
        # now=1.0, render_time=0.5 -> should be 5.0
        result = buf.evaluate(now=1.0)
        assert result == pytest.approx(5.0)

    def test_vec2_interpolation(self):
        buf = InterpolationBuffer(delay=0.0)
        buf.push((0.0, 0.0), timestamp=0.0)
        buf.push((10.0, 20.0), timestamp=1.0)
        result = buf.evaluate(now=0.5)
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(10.0)

    def test_clear(self):
        buf = InterpolationBuffer()
        buf.push(1.0, timestamp=0.0)
        buf.push(2.0, timestamp=1.0)
        buf.clear()
        assert buf.empty is True
        assert buf.sample_count == 0

    def test_sample_count(self):
        buf = InterpolationBuffer()
        assert buf.sample_count == 0
        buf.push(1.0)
        assert buf.sample_count == 1
        buf.push(2.0)
        assert buf.sample_count == 2

    def test_max_samples_eviction(self):
        buf = InterpolationBuffer(max_samples=5)
        for i in range(10):
            buf.push(float(i), timestamp=float(i))
        assert buf.sample_count == 5
