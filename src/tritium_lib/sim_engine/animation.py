# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Animation and keyframe system for Three.js smooth rendering.

Provides easing functions, keyframe interpolation, entity animation tracks,
pre-built animation library, and an interpolation buffer for smoothing
server-to-client position updates.  All output is JSON-serializable for
direct consumption by the Three.js frontend.

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Union

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Scalar = float
Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
AnimValue = Union[float, Vec2, Vec3]


# ===================================================================
# 1. Easing functions  (t in [0,1] -> t' in [0,1])
# ===================================================================

def linear(t: float) -> float:
    """No easing -- constant velocity."""
    return max(0.0, min(1.0, t))


def ease_in(t: float) -> float:
    """Quadratic ease-in -- accelerate from zero."""
    t = max(0.0, min(1.0, t))
    return t * t


def ease_out(t: float) -> float:
    """Quadratic ease-out -- decelerate to zero."""
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) * (1.0 - t)


def ease_in_out(t: float) -> float:
    """Quadratic ease-in-out -- smooth start and stop."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0


def bounce(t: float) -> float:
    """Bounce easing -- bounces at the end like a ball."""
    t = max(0.0, min(1.0, t))
    if t < 1.0 / 2.75:
        return 7.5625 * t * t
    elif t < 2.0 / 2.75:
        t -= 1.5 / 2.75
        return 7.5625 * t * t + 0.75
    elif t < 2.5 / 2.75:
        t -= 2.25 / 2.75
        return 7.5625 * t * t + 0.9375
    else:
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375


def elastic(t: float) -> float:
    """Elastic easing -- spring-like overshoot."""
    t = max(0.0, min(1.0, t))
    if t == 0.0 or t == 1.0:
        return t
    p = 0.3
    s = p / 4.0
    return math.pow(2.0, -10.0 * t) * math.sin((t - s) * (2.0 * math.pi) / p) + 1.0


def back(t: float) -> float:
    """Back easing -- slight overshoot then settle."""
    t = max(0.0, min(1.0, t))
    s = 1.70158
    return t * t * ((s + 1.0) * t - s)


# Registry of easing functions by name
EASING_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "linear": linear,
    "ease_in": ease_in,
    "ease_out": ease_out,
    "ease_in_out": ease_in_out,
    "bounce": bounce,
    "elastic": elastic,
    "back": back,
}


def get_easing(name: str) -> Callable[[float], float]:
    """Look up an easing function by name.  Defaults to linear."""
    return EASING_FUNCTIONS.get(name, linear)


# ===================================================================
# 2. Keyframe
# ===================================================================

@dataclass
class Keyframe:
    """A single animation keyframe."""

    time: float
    value: AnimValue
    easing: str = "linear"

    def to_dict(self) -> dict[str, Any]:
        v = self.value
        if isinstance(v, (tuple, list)):
            v = list(v)
        return {"time": self.time, "value": v, "easing": self.easing}


# ===================================================================
# 3. AnimationTrack
# ===================================================================

class AnimationTrack:
    """Ordered list of keyframes with time-based interpolation.

    Supports scalar floats and Vec2/Vec3 tuple interpolation.
    Keyframes are kept sorted by time.
    """

    def __init__(self, name: str = "", keyframes: list[Keyframe] | None = None) -> None:
        self.name = name
        self._keyframes: list[Keyframe] = []
        if keyframes:
            for kf in keyframes:
                self.add_keyframe(kf)

    # -- mutators --

    def add_keyframe(self, kf: Keyframe) -> None:
        """Insert a keyframe, maintaining time order."""
        self._keyframes.append(kf)
        self._keyframes.sort(key=lambda k: k.time)

    @property
    def keyframes(self) -> list[Keyframe]:
        return list(self._keyframes)

    @property
    def duration(self) -> float:
        if not self._keyframes:
            return 0.0
        return self._keyframes[-1].time - self._keyframes[0].time

    @property
    def start_time(self) -> float:
        if not self._keyframes:
            return 0.0
        return self._keyframes[0].time

    @property
    def end_time(self) -> float:
        if not self._keyframes:
            return 0.0
        return self._keyframes[-1].time

    # -- evaluation --

    def evaluate(self, t: float) -> AnimValue:
        """Interpolate the track value at time *t*.

        Before the first keyframe returns the first value.
        After the last keyframe returns the last value.
        Between two keyframes linearly interpolates using the *next*
        keyframe's easing function.
        """
        if not self._keyframes:
            return 0.0

        # Clamp: before first / after last
        if t <= self._keyframes[0].time:
            return self._keyframes[0].value
        if t >= self._keyframes[-1].time:
            return self._keyframes[-1].value

        # Find surrounding keyframes
        for i in range(len(self._keyframes) - 1):
            k0 = self._keyframes[i]
            k1 = self._keyframes[i + 1]
            if k0.time <= t <= k1.time:
                seg_dur = k1.time - k0.time
                if seg_dur <= 0:
                    return k1.value
                local_t = (t - k0.time) / seg_dur
                eased_t = get_easing(k1.easing)(local_t)
                return _interpolate(k0.value, k1.value, eased_t)

        return self._keyframes[-1].value  # pragma: no cover

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "keyframes": [kf.to_dict() for kf in self._keyframes],
            "duration": self.duration,
        }


# ===================================================================
# Interpolation helpers
# ===================================================================

def _interpolate(a: AnimValue, b: AnimValue, t: float) -> AnimValue:
    """Lerp between two values (scalar or tuple)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) + (float(b) - float(a)) * t
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return tuple(
            float(av) + (float(bv) - float(av)) * t
            for av, bv in zip(a, b)
        )
    # Fallback: return b if types mismatch
    return b  # pragma: no cover


def lerp(a: float, b: float, t: float) -> float:
    """Simple scalar lerp, exposed as a utility."""
    return a + (b - a) * t


# ===================================================================
# 4. EntityAnimation
# ===================================================================

class EntityAnimation:
    """Multiple animation tracks keyed by property name.

    Typical properties: position, rotation, scale, color, opacity.
    """

    def __init__(
        self,
        name: str = "",
        tracks: dict[str, AnimationTrack] | None = None,
        loop: bool = False,
    ) -> None:
        self.name = name
        self.tracks: dict[str, AnimationTrack] = tracks or {}
        self.loop = loop

    def add_track(self, prop: str, track: AnimationTrack) -> None:
        self.tracks[prop] = track

    @property
    def duration(self) -> float:
        if not self.tracks:
            return 0.0
        return max(t.duration for t in self.tracks.values())

    def evaluate(self, t: float) -> dict[str, AnimValue]:
        """Evaluate all tracks at time *t* and return property dict."""
        if self.loop and self.duration > 0:
            t = t % self.duration
        result: dict[str, AnimValue] = {}
        for prop, track in self.tracks.items():
            result[prop] = track.evaluate(t)
        return result

    def to_three_js(self) -> dict[str, Any]:
        """Export for Three.js AnimationClip consumption.

        Returns a dict with name, duration, loop flag, and tracks list
        matching the shape expected by the frontend Three.js adapter.
        """
        return {
            "name": self.name,
            "duration": self.duration,
            "loop": self.loop,
            "tracks": {prop: track.to_dict() for prop, track in self.tracks.items()},
        }


# ===================================================================
# 5. AnimationLibrary -- 10 pre-built animations
# ===================================================================

class AnimationLibrary:
    """Collection of pre-built animations for common sim effects."""

    @staticmethod
    def walk_cycle(duration: float = 1.0) -> EntityAnimation:
        """Looping walk cycle -- vertical bob + slight lean."""
        anim = EntityAnimation(name="walk_cycle", loop=True)
        # Vertical bob (y offset)
        bob = AnimationTrack("position_y", [
            Keyframe(0.0, 0.0, "ease_in_out"),
            Keyframe(duration * 0.25, 0.05, "ease_in_out"),
            Keyframe(duration * 0.5, 0.0, "ease_in_out"),
            Keyframe(duration * 0.75, 0.05, "ease_in_out"),
            Keyframe(duration, 0.0, "ease_in_out"),
        ])
        # Lean
        lean = AnimationTrack("rotation_z", [
            Keyframe(0.0, 0.0, "ease_in_out"),
            Keyframe(duration * 0.25, 0.03, "ease_in_out"),
            Keyframe(duration * 0.5, 0.0, "ease_in_out"),
            Keyframe(duration * 0.75, -0.03, "ease_in_out"),
            Keyframe(duration, 0.0, "ease_in_out"),
        ])
        anim.add_track("position_y", bob)
        anim.add_track("rotation_z", lean)
        return anim

    @staticmethod
    def run_cycle(duration: float = 0.6) -> EntityAnimation:
        """Looping run cycle -- larger bob, faster cadence."""
        anim = EntityAnimation(name="run_cycle", loop=True)
        bob = AnimationTrack("position_y", [
            Keyframe(0.0, 0.0, "ease_in_out"),
            Keyframe(duration * 0.25, 0.12, "ease_in_out"),
            Keyframe(duration * 0.5, 0.0, "ease_in_out"),
            Keyframe(duration * 0.75, 0.12, "ease_in_out"),
            Keyframe(duration, 0.0, "ease_in_out"),
        ])
        lean = AnimationTrack("rotation_z", [
            Keyframe(0.0, 0.0, "ease_in_out"),
            Keyframe(duration * 0.25, 0.06, "ease_in_out"),
            Keyframe(duration * 0.5, 0.0, "ease_in_out"),
            Keyframe(duration * 0.75, -0.06, "ease_in_out"),
            Keyframe(duration, 0.0, "ease_in_out"),
        ])
        anim.add_track("position_y", bob)
        anim.add_track("rotation_z", lean)
        return anim

    @staticmethod
    def death_fall(duration: float = 0.8) -> EntityAnimation:
        """One-shot death -- fall sideways, fade out."""
        anim = EntityAnimation(name="death_fall", loop=False)
        rot = AnimationTrack("rotation_z", [
            Keyframe(0.0, 0.0, "ease_in"),
            Keyframe(duration * 0.6, math.pi / 2, "bounce"),
            Keyframe(duration, math.pi / 2, "linear"),
        ])
        opacity = AnimationTrack("opacity", [
            Keyframe(0.0, 1.0, "linear"),
            Keyframe(duration * 0.7, 1.0, "linear"),
            Keyframe(duration, 0.0, "ease_out"),
        ])
        pos_y = AnimationTrack("position_y", [
            Keyframe(0.0, 0.0, "ease_in"),
            Keyframe(duration * 0.6, -0.5, "bounce"),
            Keyframe(duration, -0.5, "linear"),
        ])
        anim.add_track("rotation_z", rot)
        anim.add_track("opacity", opacity)
        anim.add_track("position_y", pos_y)
        return anim

    @staticmethod
    def explosion_shake(duration: float = 0.5) -> EntityAnimation:
        """One-shot camera/object shake from explosion."""
        anim = EntityAnimation(name="explosion_shake", loop=False)
        n_shakes = 8
        kfs_x: list[Keyframe] = [Keyframe(0.0, 0.0, "linear")]
        kfs_y: list[Keyframe] = [Keyframe(0.0, 0.0, "linear")]
        for i in range(1, n_shakes + 1):
            t = duration * i / (n_shakes + 1)
            decay = 1.0 - (i / (n_shakes + 1))
            amp = 0.15 * decay
            sign = 1.0 if i % 2 == 0 else -1.0
            kfs_x.append(Keyframe(t, sign * amp, "linear"))
            kfs_y.append(Keyframe(t, -sign * amp * 0.7, "linear"))
        kfs_x.append(Keyframe(duration, 0.0, "ease_out"))
        kfs_y.append(Keyframe(duration, 0.0, "ease_out"))
        anim.add_track("offset_x", AnimationTrack("offset_x", kfs_x))
        anim.add_track("offset_y", AnimationTrack("offset_y", kfs_y))
        # Scale pulse
        scale = AnimationTrack("scale", [
            Keyframe(0.0, 1.0, "linear"),
            Keyframe(duration * 0.1, 1.3, "ease_out"),
            Keyframe(duration, 1.0, "ease_out"),
        ])
        anim.add_track("scale", scale)
        return anim

    @staticmethod
    def muzzle_flash(duration: float = 0.12) -> EntityAnimation:
        """Very short bright flash for weapon fire."""
        anim = EntityAnimation(name="muzzle_flash", loop=False)
        opacity = AnimationTrack("opacity", [
            Keyframe(0.0, 0.0, "linear"),
            Keyframe(duration * 0.1, 1.0, "linear"),
            Keyframe(duration, 0.0, "ease_out"),
        ])
        scale = AnimationTrack("scale", [
            Keyframe(0.0, 0.5, "linear"),
            Keyframe(duration * 0.15, 1.5, "ease_out"),
            Keyframe(duration, 0.2, "ease_out"),
        ])
        anim.add_track("opacity", opacity)
        anim.add_track("scale", scale)
        return anim

    @staticmethod
    def vehicle_bounce(duration: float = 0.8) -> EntityAnimation:
        """Looping suspension bounce for moving vehicles."""
        anim = EntityAnimation(name="vehicle_bounce", loop=True)
        bob = AnimationTrack("position_y", [
            Keyframe(0.0, 0.0, "ease_in_out"),
            Keyframe(duration * 0.5, 0.03, "ease_in_out"),
            Keyframe(duration, 0.0, "ease_in_out"),
        ])
        pitch = AnimationTrack("rotation_x", [
            Keyframe(0.0, 0.0, "ease_in_out"),
            Keyframe(duration * 0.25, 0.01, "ease_in_out"),
            Keyframe(duration * 0.75, -0.01, "ease_in_out"),
            Keyframe(duration, 0.0, "ease_in_out"),
        ])
        anim.add_track("position_y", bob)
        anim.add_track("rotation_x", pitch)
        return anim

    @staticmethod
    def helicopter_hover(duration: float = 2.0) -> EntityAnimation:
        """Looping hover drift -- slow vertical + horizontal sway."""
        anim = EntityAnimation(name="helicopter_hover", loop=True)
        bob = AnimationTrack("position_y", [
            Keyframe(0.0, 0.0, "ease_in_out"),
            Keyframe(duration * 0.5, 0.15, "ease_in_out"),
            Keyframe(duration, 0.0, "ease_in_out"),
        ])
        sway = AnimationTrack("position_x", [
            Keyframe(0.0, 0.0, "ease_in_out"),
            Keyframe(duration * 0.25, 0.08, "ease_in_out"),
            Keyframe(duration * 0.75, -0.08, "ease_in_out"),
            Keyframe(duration, 0.0, "ease_in_out"),
        ])
        tilt = AnimationTrack("rotation_z", [
            Keyframe(0.0, 0.0, "ease_in_out"),
            Keyframe(duration * 0.25, 0.02, "ease_in_out"),
            Keyframe(duration * 0.75, -0.02, "ease_in_out"),
            Keyframe(duration, 0.0, "ease_in_out"),
        ])
        anim.add_track("position_y", bob)
        anim.add_track("position_x", sway)
        anim.add_track("rotation_z", tilt)
        return anim

    @staticmethod
    def flag_wave(duration: float = 2.5) -> EntityAnimation:
        """Looping cloth wave for flags and banners."""
        anim = EntityAnimation(name="flag_wave", loop=True)
        wave = AnimationTrack("deform_x", [
            Keyframe(0.0, 0.0, "ease_in_out"),
            Keyframe(duration * 0.25, 0.1, "ease_in_out"),
            Keyframe(duration * 0.5, 0.0, "ease_in_out"),
            Keyframe(duration * 0.75, -0.08, "ease_in_out"),
            Keyframe(duration, 0.0, "ease_in_out"),
        ])
        flutter = AnimationTrack("deform_y", [
            Keyframe(0.0, 0.0, "ease_in_out"),
            Keyframe(duration * 0.33, 0.04, "ease_in_out"),
            Keyframe(duration * 0.66, -0.03, "ease_in_out"),
            Keyframe(duration, 0.0, "ease_in_out"),
        ])
        anim.add_track("deform_x", wave)
        anim.add_track("deform_y", flutter)
        return anim

    @staticmethod
    def fire_flicker(duration: float = 0.4) -> EntityAnimation:
        """Looping fire flicker -- scale + opacity jitter."""
        anim = EntityAnimation(name="fire_flicker", loop=True)
        scale = AnimationTrack("scale", [
            Keyframe(0.0, 1.0, "linear"),
            Keyframe(duration * 0.2, 1.15, "linear"),
            Keyframe(duration * 0.4, 0.9, "linear"),
            Keyframe(duration * 0.6, 1.1, "linear"),
            Keyframe(duration * 0.8, 0.95, "linear"),
            Keyframe(duration, 1.0, "linear"),
        ])
        opacity = AnimationTrack("opacity", [
            Keyframe(0.0, 0.9, "linear"),
            Keyframe(duration * 0.3, 1.0, "linear"),
            Keyframe(duration * 0.5, 0.8, "linear"),
            Keyframe(duration * 0.7, 1.0, "linear"),
            Keyframe(duration, 0.9, "linear"),
        ])
        anim.add_track("scale", scale)
        anim.add_track("opacity", opacity)
        return anim

    @staticmethod
    def smoke_rise(duration: float = 3.0) -> EntityAnimation:
        """One-shot smoke plume -- rises, expands, fades."""
        anim = EntityAnimation(name="smoke_rise", loop=False)
        pos_y = AnimationTrack("position_y", [
            Keyframe(0.0, 0.0, "ease_out"),
            Keyframe(duration, 2.0, "ease_out"),
        ])
        scale = AnimationTrack("scale", [
            Keyframe(0.0, 0.3, "ease_out"),
            Keyframe(duration * 0.5, 1.0, "ease_out"),
            Keyframe(duration, 1.8, "ease_out"),
        ])
        opacity = AnimationTrack("opacity", [
            Keyframe(0.0, 0.8, "linear"),
            Keyframe(duration * 0.3, 0.7, "linear"),
            Keyframe(duration, 0.0, "ease_in"),
        ])
        anim.add_track("position_y", pos_y)
        anim.add_track("scale", scale)
        anim.add_track("opacity", opacity)
        return anim

    # -- registry access --

    _REGISTRY: dict[str, Callable[..., EntityAnimation]] | None = None

    @classmethod
    def _build_registry(cls) -> dict[str, Callable[..., EntityAnimation]]:
        if cls._REGISTRY is None:
            cls._REGISTRY = {
                "walk_cycle": cls.walk_cycle,
                "run_cycle": cls.run_cycle,
                "death_fall": cls.death_fall,
                "explosion_shake": cls.explosion_shake,
                "muzzle_flash": cls.muzzle_flash,
                "vehicle_bounce": cls.vehicle_bounce,
                "helicopter_hover": cls.helicopter_hover,
                "flag_wave": cls.flag_wave,
                "fire_flicker": cls.fire_flicker,
                "smoke_rise": cls.smoke_rise,
            }
        return cls._REGISTRY

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> EntityAnimation | None:
        """Retrieve a pre-built animation by name, or None if unknown."""
        reg = cls._build_registry()
        factory = reg.get(name)
        if factory is None:
            return None
        return factory(**kwargs)

    @classmethod
    def list_animations(cls) -> list[str]:
        """Return names of all available pre-built animations."""
        return sorted(cls._build_registry().keys())


# ===================================================================
# 6. InterpolationBuffer
# ===================================================================

@dataclass
class _BufferSample:
    """Single timestamped position sample."""
    timestamp: float
    value: AnimValue


class InterpolationBuffer:
    """Smooths server -> client position updates.

    Stores a short history of server positions and interpolates between
    them with a configurable delay, preventing jittery movement when
    network updates arrive at irregular intervals.

    Parameters
    ----------
    delay : float
        How far behind real-time to render (seconds).  Higher = smoother
        but more latency.  Typical: 0.05 -- 0.2.
    max_samples : int
        Maximum stored samples before oldest are discarded.
    """

    def __init__(self, delay: float = 0.1, max_samples: int = 30) -> None:
        self.delay = delay
        self.max_samples = max_samples
        self._samples: deque[_BufferSample] = deque(maxlen=max_samples)

    def push(self, value: AnimValue, timestamp: float | None = None) -> None:
        """Add a new server sample."""
        ts = timestamp if timestamp is not None else time.monotonic()
        self._samples.append(_BufferSample(timestamp=ts, value=value))

    def evaluate(self, now: float | None = None) -> AnimValue | None:
        """Get the smoothed value at the current render time.

        Returns None if fewer than 2 samples are available.
        """
        if len(self._samples) < 2:
            if len(self._samples) == 1:
                return self._samples[0].value
            return None

        render_time = (now if now is not None else time.monotonic()) - self.delay

        # Before all samples
        if render_time <= self._samples[0].timestamp:
            return self._samples[0].value

        # After all samples -- extrapolate from last two
        if render_time >= self._samples[-1].timestamp:
            return self._samples[-1].value

        # Find bracketing samples
        for i in range(len(self._samples) - 1):
            s0 = self._samples[i]
            s1 = self._samples[i + 1]
            if s0.timestamp <= render_time <= s1.timestamp:
                seg = s1.timestamp - s0.timestamp
                if seg <= 0:
                    return s1.value
                t = (render_time - s0.timestamp) / seg
                return _interpolate(s0.value, s1.value, t)

        return self._samples[-1].value  # pragma: no cover

    def clear(self) -> None:
        """Discard all buffered samples."""
        self._samples.clear()

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def empty(self) -> bool:
        return len(self._samples) == 0
