# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine.animation — easing, keyframes, tracks, library, buffer."""

from __future__ import annotations

import math
import time

import pytest

from tritium_lib.sim_engine.animation import (
    EASING_FUNCTIONS,
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


# ===================================================================
# Easing functions
# ===================================================================

class TestEasingFunctions:
    """All easing functions: boundary values and monotonicity."""

    @pytest.mark.parametrize("fn", [linear, ease_in, ease_out, ease_in_out, bounce, elastic, back])
    def test_easing_at_zero(self, fn):
        assert fn(0.0) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("fn", [linear, ease_in, ease_out, ease_in_out, bounce, elastic])
    def test_easing_at_one(self, fn):
        assert fn(1.0) == pytest.approx(1.0, abs=1e-9)

    def test_linear_midpoint(self):
        assert linear(0.5) == pytest.approx(0.5)

    def test_ease_in_slower_start(self):
        # At t=0.5, ease_in should be below 0.5 (accelerating)
        assert ease_in(0.5) < 0.5

    def test_ease_out_faster_start(self):
        # At t=0.5, ease_out should be above 0.5 (decelerating)
        assert ease_out(0.5) > 0.5

    def test_ease_in_out_midpoint(self):
        assert ease_in_out(0.5) == pytest.approx(0.5, abs=0.01)

    def test_bounce_stays_in_range(self):
        for i in range(11):
            t = i / 10.0
            v = bounce(t)
            assert 0.0 <= v <= 1.0 + 1e-9, f"bounce({t}) = {v}"

    def test_elastic_overshoots(self):
        # Elastic can exceed 1.0 mid-animation
        vals = [elastic(i / 20.0) for i in range(21)]
        assert any(v > 1.0 for v in vals), "elastic should overshoot"

    def test_back_goes_negative(self):
        # Back easing dips below 0 early on
        vals = [back(i / 20.0) for i in range(21)]
        assert any(v < 0.0 for v in vals), "back should go negative"

    def test_clamping_below_zero(self):
        for fn in [linear, ease_in, ease_out, ease_in_out, bounce, elastic, back]:
            v = fn(-0.5)
            assert v == fn(0.0)

    def test_clamping_above_one(self):
        for fn in [linear, ease_in, ease_out, ease_in_out, bounce, elastic, back]:
            v = fn(1.5)
            assert v == fn(1.0)

    def test_easing_registry_has_all(self):
        assert set(EASING_FUNCTIONS.keys()) == {
            "linear", "ease_in", "ease_out", "ease_in_out",
            "bounce", "elastic", "back",
        }

    def test_get_easing_known(self):
        assert get_easing("bounce") is bounce

    def test_get_easing_unknown_returns_linear(self):
        assert get_easing("nonexistent") is linear


# ===================================================================
# Keyframe
# ===================================================================

class TestKeyframe:
    def test_basic_creation(self):
        kf = Keyframe(time=0.5, value=1.0)
        assert kf.time == 0.5
        assert kf.value == 1.0
        assert kf.easing == "linear"

    def test_custom_easing(self):
        kf = Keyframe(time=1.0, value=(1.0, 2.0), easing="bounce")
        assert kf.easing == "bounce"

    def test_to_dict_scalar(self):
        kf = Keyframe(time=0.0, value=0.5, easing="ease_in")
        d = kf.to_dict()
        assert d == {"time": 0.0, "value": 0.5, "easing": "ease_in"}

    def test_to_dict_tuple(self):
        kf = Keyframe(time=1.0, value=(1.0, 2.0))
        d = kf.to_dict()
        assert d["value"] == [1.0, 2.0]

    def test_to_dict_vec3(self):
        kf = Keyframe(time=0.0, value=(1.0, 2.0, 3.0))
        d = kf.to_dict()
        assert d["value"] == [1.0, 2.0, 3.0]


# ===================================================================
# AnimationTrack
# ===================================================================

class TestAnimationTrack:
    def test_empty_track_returns_zero(self):
        track = AnimationTrack("empty")
        assert track.evaluate(0.0) == 0.0
        assert track.duration == 0.0

    def test_single_keyframe(self):
        track = AnimationTrack("test", [Keyframe(0.0, 5.0)])
        assert track.evaluate(0.0) == 5.0
        assert track.evaluate(1.0) == 5.0

    def test_two_keyframes_linear(self):
        track = AnimationTrack("x", [
            Keyframe(0.0, 0.0),
            Keyframe(1.0, 10.0, "linear"),
        ])
        assert track.evaluate(0.0) == pytest.approx(0.0)
        assert track.evaluate(0.5) == pytest.approx(5.0)
        assert track.evaluate(1.0) == pytest.approx(10.0)

    def test_clamp_before_first(self):
        track = AnimationTrack("x", [
            Keyframe(1.0, 5.0),
            Keyframe(2.0, 10.0),
        ])
        assert track.evaluate(0.0) == 5.0

    def test_clamp_after_last(self):
        track = AnimationTrack("x", [
            Keyframe(0.0, 0.0),
            Keyframe(1.0, 10.0),
        ])
        assert track.evaluate(5.0) == 10.0

    def test_three_keyframes(self):
        track = AnimationTrack("x", [
            Keyframe(0.0, 0.0),
            Keyframe(1.0, 10.0, "linear"),
            Keyframe(2.0, 0.0, "linear"),
        ])
        assert track.evaluate(0.5) == pytest.approx(5.0)
        assert track.evaluate(1.0) == pytest.approx(10.0)
        assert track.evaluate(1.5) == pytest.approx(5.0)

    def test_vec2_interpolation(self):
        track = AnimationTrack("pos", [
            Keyframe(0.0, (0.0, 0.0)),
            Keyframe(1.0, (10.0, 20.0), "linear"),
        ])
        val = track.evaluate(0.5)
        assert isinstance(val, tuple)
        assert len(val) == 2
        assert val[0] == pytest.approx(5.0)
        assert val[1] == pytest.approx(10.0)

    def test_vec3_interpolation(self):
        track = AnimationTrack("pos", [
            Keyframe(0.0, (0.0, 0.0, 0.0)),
            Keyframe(1.0, (3.0, 6.0, 9.0), "linear"),
        ])
        val = track.evaluate(0.5)
        assert len(val) == 3
        assert val[2] == pytest.approx(4.5)

    def test_ease_in_easing_applied(self):
        track = AnimationTrack("x", [
            Keyframe(0.0, 0.0),
            Keyframe(1.0, 10.0, "ease_in"),
        ])
        # At t=0.5, ease_in(0.5) = 0.25, so value should be 2.5
        assert track.evaluate(0.5) == pytest.approx(2.5)

    def test_keyframes_sorted_on_add(self):
        track = AnimationTrack("x")
        track.add_keyframe(Keyframe(2.0, 20.0))
        track.add_keyframe(Keyframe(0.0, 0.0))
        track.add_keyframe(Keyframe(1.0, 10.0))
        assert [kf.time for kf in track.keyframes] == [0.0, 1.0, 2.0]

    def test_duration(self):
        track = AnimationTrack("x", [
            Keyframe(1.0, 0.0),
            Keyframe(3.0, 10.0),
        ])
        assert track.duration == 2.0

    def test_start_end_time(self):
        track = AnimationTrack("x", [
            Keyframe(0.5, 0.0),
            Keyframe(2.5, 10.0),
        ])
        assert track.start_time == 0.5
        assert track.end_time == 2.5

    def test_to_dict(self):
        track = AnimationTrack("alpha", [
            Keyframe(0.0, 0.0),
            Keyframe(1.0, 1.0),
        ])
        d = track.to_dict()
        assert d["name"] == "alpha"
        assert len(d["keyframes"]) == 2
        assert d["duration"] == 1.0


# ===================================================================
# EntityAnimation
# ===================================================================

class TestEntityAnimation:
    def test_basic_evaluate(self):
        anim = EntityAnimation(name="test")
        anim.add_track("opacity", AnimationTrack("opacity", [
            Keyframe(0.0, 1.0),
            Keyframe(1.0, 0.0, "linear"),
        ]))
        result = anim.evaluate(0.5)
        assert "opacity" in result
        assert result["opacity"] == pytest.approx(0.5)

    def test_multiple_tracks(self):
        anim = EntityAnimation(name="test")
        anim.add_track("x", AnimationTrack("x", [
            Keyframe(0.0, 0.0), Keyframe(1.0, 10.0, "linear"),
        ]))
        anim.add_track("y", AnimationTrack("y", [
            Keyframe(0.0, 0.0), Keyframe(1.0, 20.0, "linear"),
        ]))
        result = anim.evaluate(0.5)
        assert result["x"] == pytest.approx(5.0)
        assert result["y"] == pytest.approx(10.0)

    def test_loop_wraps_time(self):
        anim = EntityAnimation(name="loop_test", loop=True)
        anim.add_track("v", AnimationTrack("v", [
            Keyframe(0.0, 0.0), Keyframe(1.0, 10.0, "linear"),
        ]))
        # t=1.5 with loop and duration=1.0 should wrap to t=0.5
        result = anim.evaluate(1.5)
        assert result["v"] == pytest.approx(5.0)

    def test_no_loop_clamps(self):
        anim = EntityAnimation(name="no_loop", loop=False)
        anim.add_track("v", AnimationTrack("v", [
            Keyframe(0.0, 0.0), Keyframe(1.0, 10.0, "linear"),
        ]))
        result = anim.evaluate(2.0)
        assert result["v"] == pytest.approx(10.0)

    def test_duration_max_of_tracks(self):
        anim = EntityAnimation(name="test")
        anim.add_track("a", AnimationTrack("a", [
            Keyframe(0.0, 0.0), Keyframe(1.0, 1.0),
        ]))
        anim.add_track("b", AnimationTrack("b", [
            Keyframe(0.0, 0.0), Keyframe(3.0, 1.0),
        ]))
        assert anim.duration == 3.0

    def test_empty_duration(self):
        anim = EntityAnimation()
        assert anim.duration == 0.0

    def test_to_three_js_structure(self):
        anim = EntityAnimation(name="clip1", loop=True)
        anim.add_track("opacity", AnimationTrack("opacity", [
            Keyframe(0.0, 1.0), Keyframe(1.0, 0.0),
        ]))
        out = anim.to_three_js()
        assert out["name"] == "clip1"
        assert out["loop"] is True
        assert "opacity" in out["tracks"]
        assert isinstance(out["duration"], float)

    def test_to_three_js_has_keyframes(self):
        anim = EntityAnimation(name="test")
        anim.add_track("scale", AnimationTrack("scale", [
            Keyframe(0.0, 1.0), Keyframe(0.5, 2.0), Keyframe(1.0, 1.0),
        ]))
        out = anim.to_three_js()
        kfs = out["tracks"]["scale"]["keyframes"]
        assert len(kfs) == 3
        assert kfs[1]["value"] == 2.0


# ===================================================================
# AnimationLibrary
# ===================================================================

class TestAnimationLibrary:
    ALL_NAMES = [
        "walk_cycle", "run_cycle", "death_fall", "explosion_shake",
        "muzzle_flash", "vehicle_bounce", "helicopter_hover",
        "flag_wave", "fire_flicker", "smoke_rise",
    ]

    def test_list_animations_has_ten(self):
        names = AnimationLibrary.list_animations()
        assert len(names) == 10

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_get_returns_entity_animation(self, name):
        anim = AnimationLibrary.get(name)
        assert isinstance(anim, EntityAnimation)

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_animation_has_tracks(self, name):
        anim = AnimationLibrary.get(name)
        assert len(anim.tracks) > 0

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_animation_name_matches(self, name):
        anim = AnimationLibrary.get(name)
        assert anim.name == name

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_animation_evaluates_at_zero(self, name):
        anim = AnimationLibrary.get(name)
        result = anim.evaluate(0.0)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_animation_to_three_js(self, name):
        anim = AnimationLibrary.get(name)
        out = anim.to_three_js()
        assert "tracks" in out
        assert "duration" in out
        assert out["name"] == name

    def test_get_unknown_returns_none(self):
        assert AnimationLibrary.get("nonexistent") is None

    def test_walk_cycle_loops(self):
        anim = AnimationLibrary.get("walk_cycle")
        assert anim.loop is True

    def test_death_fall_no_loop(self):
        anim = AnimationLibrary.get("death_fall")
        assert anim.loop is False

    def test_smoke_rise_no_loop(self):
        anim = AnimationLibrary.get("smoke_rise")
        assert anim.loop is False

    def test_fire_flicker_loops(self):
        anim = AnimationLibrary.get("fire_flicker")
        assert anim.loop is True

    def test_custom_duration(self):
        anim = AnimationLibrary.get("walk_cycle", duration=2.0)
        assert anim.duration == pytest.approx(2.0, abs=0.01)

    def test_explosion_shake_has_offset_tracks(self):
        anim = AnimationLibrary.get("explosion_shake")
        assert "offset_x" in anim.tracks
        assert "offset_y" in anim.tracks
        assert "scale" in anim.tracks

    def test_muzzle_flash_short_duration(self):
        anim = AnimationLibrary.get("muzzle_flash")
        assert anim.duration < 0.5

    def test_helicopter_hover_has_sway(self):
        anim = AnimationLibrary.get("helicopter_hover")
        assert "position_x" in anim.tracks


# ===================================================================
# InterpolationBuffer
# ===================================================================

class TestInterpolationBuffer:
    def test_empty_returns_none(self):
        buf = InterpolationBuffer()
        assert buf.evaluate(now=0.0) is None
        assert buf.empty

    def test_single_sample_returns_value(self):
        buf = InterpolationBuffer(delay=0.1)
        buf.push(5.0, timestamp=1.0)
        assert buf.evaluate(now=1.2) == 5.0

    def test_two_samples_interpolate(self):
        buf = InterpolationBuffer(delay=0.0)
        buf.push(0.0, timestamp=0.0)
        buf.push(10.0, timestamp=1.0)
        val = buf.evaluate(now=0.5)
        assert val == pytest.approx(5.0)

    def test_delay_offsets_render_time(self):
        buf = InterpolationBuffer(delay=0.5)
        buf.push(0.0, timestamp=0.0)
        buf.push(10.0, timestamp=1.0)
        # now=1.0 minus delay=0.5 -> render_time=0.5
        val = buf.evaluate(now=1.0)
        assert val == pytest.approx(5.0)

    def test_before_all_samples_returns_first(self):
        buf = InterpolationBuffer(delay=0.0)
        buf.push(5.0, timestamp=1.0)
        buf.push(10.0, timestamp=2.0)
        val = buf.evaluate(now=0.5)
        assert val == 5.0

    def test_after_all_samples_returns_last(self):
        buf = InterpolationBuffer(delay=0.0)
        buf.push(5.0, timestamp=1.0)
        buf.push(10.0, timestamp=2.0)
        val = buf.evaluate(now=5.0)
        assert val == 10.0

    def test_vec2_interpolation(self):
        buf = InterpolationBuffer(delay=0.0)
        buf.push((0.0, 0.0), timestamp=0.0)
        buf.push((10.0, 20.0), timestamp=1.0)
        val = buf.evaluate(now=0.5)
        assert val[0] == pytest.approx(5.0)
        assert val[1] == pytest.approx(10.0)

    def test_clear(self):
        buf = InterpolationBuffer()
        buf.push(1.0, timestamp=0.0)
        buf.push(2.0, timestamp=1.0)
        assert buf.sample_count == 2
        buf.clear()
        assert buf.sample_count == 0
        assert buf.empty

    def test_max_samples_eviction(self):
        buf = InterpolationBuffer(max_samples=5)
        for i in range(10):
            buf.push(float(i), timestamp=float(i))
        assert buf.sample_count == 5

    def test_sample_count(self):
        buf = InterpolationBuffer()
        assert buf.sample_count == 0
        buf.push(1.0, timestamp=0.0)
        assert buf.sample_count == 1

    def test_multiple_segments(self):
        buf = InterpolationBuffer(delay=0.0)
        buf.push(0.0, timestamp=0.0)
        buf.push(10.0, timestamp=1.0)
        buf.push(10.0, timestamp=2.0)
        buf.push(0.0, timestamp=3.0)
        # t=2.5 should be between 10.0 and 0.0 -> 5.0
        val = buf.evaluate(now=2.5)
        assert val == pytest.approx(5.0)


# ===================================================================
# lerp utility
# ===================================================================

class TestLerp:
    def test_lerp_zero(self):
        assert lerp(0.0, 10.0, 0.0) == 0.0

    def test_lerp_one(self):
        assert lerp(0.0, 10.0, 1.0) == 10.0

    def test_lerp_mid(self):
        assert lerp(5.0, 15.0, 0.5) == 10.0

    def test_lerp_extrapolate(self):
        assert lerp(0.0, 10.0, 2.0) == 20.0
