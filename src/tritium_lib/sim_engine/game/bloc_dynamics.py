# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""BlocDynamicsTracker -- gives the three-way rival-faction street war a voice.

A rival-faction riot has THREE parties fighting at once: bloc A, bloc B, and
the police containing both.  The single-crowd narration layer (crowd_event
beats in ``riot_police.py`` / the engine's ``_emit_crowd_beats``) speaks for
the crowd-vs-police arc, but says nothing about the red-vs-cyan dynamic the
operator is watching on the tinted map.  This tracker fills that gap.

It is fed the LIVE per-bloc fighting strength each tick (a plain
``{faction_id: violent_count}`` dict the engine already computes for the HUD)
plus the police kettle target, and emits ONE ``bloc_event`` beat per
meaningful transition -- never per-tick spam.  A static-phrase announcer
(``WarAnnouncer._on_bloc_event``, LLM-free banks) turns those beats into
callouts and degrades to silence when a beat is unknown or missing.

Beats (carried in ``data["beat"]``)
-----------------------------------
  * ``blocs_clash``   -- once, when 2+ blocs are BOTH violent at/above
                         ``clash_min`` in the same tick: the mobs have met and
                         are fighting each other ("RED and CYAN clashing!").
  * ``bloc_kettled``  -- once per distinct kettle target: the police cordoned a
                         NAMED bloc, so the other bloc is left to scatter
                         ("RED bloc kettled -- CYAN scattering!").  Re-arms when
                         the kettle target changes or clears.
  * ``bloc_dominant`` -- when one bloc's strength decisively overtakes the
                         other (``dominance_ratio`` and ``dominance_margin``):
                         the balance of the street shifted.  Re-fires only when
                         dominance FLIPS to the other bloc, and no more often
                         than ``dominant_cooldown`` seconds.

Every beat carries a stable, template-friendly payload::

    {"beat": <str>,
     "a_id": <faction id>, "a_name": <display name>, "a_count": <int>,
     "b_id": <faction id>, "b_name": <display name>, "b_count": <int>}

For ``bloc_kettled`` the ``a_*`` fields are the KETTLED bloc and ``b_*`` the
rival; for ``blocs_clash`` / ``bloc_dominant`` ``a_*`` is the stronger bloc.

Graceful degradation
---------------------
With fewer than two known blocs the tracker is inert (no beats) -- a
single-faction riot never triggers it, so the byte-identical legacy path is
untouched.  A publish exception is swallowed; narration never breaks the tick.

This is a stand-in narrator (deterministic thresholds on counts), NOT Graphling
cognition.
"""

from __future__ import annotations

from typing import Any


# Default thresholds (tunable per instance).
_CLASH_MIN: int = 3            # both blocs violent >= this -> the mobs have met
_DOMINANCE_RATIO: float = 1.6  # stronger/weaker strength ratio for dominance
_DOMINANCE_MARGIN: int = 3     # AND absolute strength lead for dominance
_DOMINANT_COOLDOWN: float = 8.0  # min seconds between bloc_dominant beats


class BlocDynamicsTracker:
    """Stand-in narrator for the red-vs-cyan-vs-police three-way dynamic."""

    def __init__(
        self,
        event_bus: Any,
        faction_specs: dict[str, dict[str, str]],
        *,
        clash_min: int = _CLASH_MIN,
        dominance_ratio: float = _DOMINANCE_RATIO,
        dominance_margin: int = _DOMINANCE_MARGIN,
        dominant_cooldown: float = _DOMINANT_COOLDOWN,
    ) -> None:
        """``faction_specs`` maps faction_id -> {"name": str, "color": str}."""
        self._event_bus = event_bus
        self._specs = dict(faction_specs or {})
        self._clash_min = clash_min
        self._dominance_ratio = dominance_ratio
        self._dominance_margin = dominance_margin
        self._dominant_cooldown = dominant_cooldown

        self._clock: float = 0.0
        self._clash_announced: bool = False
        self._kettled_announced: str | None = None   # last kettled bloc voiced
        self._dominant_announced: str | None = None   # last dominant bloc voiced
        self._last_dominant_time: float = -1e9

    # -- Public state ----------------------------------------------------------

    @property
    def active(self) -> bool:
        """True only with 2+ known blocs (else the tracker is inert)."""
        return len(self._specs) >= 2

    # -- Tick ------------------------------------------------------------------

    def update(
        self,
        dt: float,
        counts: dict[str, int],
        kettled_faction: str | None = None,
    ) -> None:
        """Advance the narrator one tick.

        ``counts`` is ``{faction_id: violent_count}`` this tick (unknown blocs
        ignored); ``kettled_faction`` is the police squad's current kettle
        target (``None`` when not kettling a specific bloc).  Emits at most a
        few beats, each guarded so it never repeats without a real transition.
        """
        if not self.active:
            return
        self._clock += dt

        # Restrict to known blocs, defaulting missing ones to 0 strength.
        strengths = {fid: int(counts.get(fid, 0)) for fid in self._specs}
        # The two strongest blocs (deterministic tiebreak by id).
        ordered = sorted(
            strengths.items(), key=lambda kv: (-kv[1], kv[0])
        )
        (a_id, a_n), (b_id, b_n) = ordered[0], ordered[1]

        self._maybe_clash(a_id, a_n, b_id, b_n)
        self._maybe_kettled(kettled_faction, strengths, a_id, a_n, b_id, b_n)
        self._maybe_dominant(a_id, a_n, b_id, b_n)

    # -- Beat rules ------------------------------------------------------------

    def _maybe_clash(self, a_id, a_n, b_id, b_n) -> None:
        if self._clash_announced:
            return
        if a_n >= self._clash_min and b_n >= self._clash_min:
            self._clash_announced = True
            self._emit("blocs_clash", a_id, a_n, b_id, b_n)

    def _maybe_kettled(self, kettled, strengths, a_id, a_n, b_id, b_n) -> None:
        if kettled is None or kettled not in self._specs:
            # Kettle lifted -> re-arm so the NEXT kettle re-announces.
            self._kettled_announced = None
            return
        if kettled == self._kettled_announced:
            return
        self._kettled_announced = kettled
        # a_* = the kettled bloc; b_* = the strongest OTHER bloc.
        k_n = strengths.get(kettled, 0)
        others = sorted(
            ((fid, n) for fid, n in strengths.items() if fid != kettled),
            key=lambda kv: (-kv[1], kv[0]),
        )
        o_id, o_n = others[0] if others else (b_id, b_n)
        self._emit("bloc_kettled", kettled, k_n, o_id, o_n)

    def _maybe_dominant(self, a_id, a_n, b_id, b_n) -> None:
        # Decisive lead: ratio AND absolute margin, and both non-trivial.
        if a_n < self._clash_min:
            return
        if not (a_n >= b_n * self._dominance_ratio
                and a_n - b_n >= self._dominance_margin):
            return
        if a_id == self._dominant_announced:
            return
        if self._clock - self._last_dominant_time < self._dominant_cooldown:
            return
        self._dominant_announced = a_id
        self._last_dominant_time = self._clock
        self._emit("bloc_dominant", a_id, a_n, b_id, b_n)

    # -- Emit ------------------------------------------------------------------

    def _emit(self, beat: str, a_id, a_n, b_id, b_n) -> None:
        a_spec = self._specs.get(a_id, {})
        b_spec = self._specs.get(b_id, {})
        try:
            self._event_bus.publish("bloc_event", {
                "beat": beat,
                "a_id": a_id,
                "a_name": a_spec.get("name", a_id),
                "a_count": int(a_n),
                "b_id": b_id,
                "b_name": b_spec.get("name", b_id),
                "b_count": int(b_n),
            })
        except Exception:
            pass

    # -- Lifecycle -------------------------------------------------------------

    def reset(self) -> None:
        """Clear all beat guards (new riot starts silent)."""
        self._clock = 0.0
        self._clash_announced = False
        self._kettled_announced = None
        self._dominant_announced = None
        self._last_dominant_time = -1e9
