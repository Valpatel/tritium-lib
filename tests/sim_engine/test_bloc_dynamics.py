# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""BlocDynamicsTracker -- the three-way rival-faction narration beats.

Pins the LLM-free stand-in narrator that voices the red-vs-cyan-vs-police
dynamic: blocs_clash (the mobs meet), bloc_kettled (police cordon one bloc),
bloc_dominant (the balance shifts).  Beats fire ONCE per transition, degrade to
silence with <2 blocs, and never crash the tick.
"""

from __future__ import annotations

from tritium_lib.sim_engine.game.bloc_dynamics import BlocDynamicsTracker


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish(self, topic: str, data: dict) -> None:
        self.events.append((topic, data))

    def bloc(self, beat: str | None = None) -> list[dict]:
        out = [d for t, d in self.events if t == "bloc_event"]
        if beat is not None:
            out = [d for d in out if d.get("beat") == beat]
        return out


_SPECS = {
    "red_bloc": {"name": "Populist Front", "color": "#ff2a6d"},
    "cyan_bloc": {"name": "Civic Union", "color": "#00f0ff"},
}


def _tracker(**kw) -> tuple[_FakeBus, BlocDynamicsTracker]:
    bus = _FakeBus()
    return bus, BlocDynamicsTracker(bus, _SPECS, **kw)


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


def test_single_bloc_is_inert():
    bus = _FakeBus()
    tr = BlocDynamicsTracker(bus, {"red_bloc": {"name": "Red", "color": "#f00"}})
    assert tr.active is False
    tr.update(0.1, {"red_bloc": 10})
    assert bus.bloc() == []


def test_no_specs_is_inert():
    bus = _FakeBus()
    tr = BlocDynamicsTracker(bus, {})
    assert tr.active is False
    tr.update(0.1, {"red_bloc": 10, "cyan_bloc": 10})
    assert bus.bloc() == []


def test_publish_exception_never_propagates():
    class _Boom:
        def publish(self, *a, **k):
            raise RuntimeError("bus down")

    tr = BlocDynamicsTracker(_Boom(), _SPECS)
    # Must not raise even though every emit blows up.
    tr.update(0.1, {"red_bloc": 6, "cyan_bloc": 6})


# ---------------------------------------------------------------------------
# blocs_clash
# ---------------------------------------------------------------------------


def test_clash_fires_once_when_both_blocs_violent():
    bus, tr = _tracker()
    # Below the clash floor on one side -> no clash yet.
    tr.update(0.1, {"red_bloc": 5, "cyan_bloc": 2})
    assert bus.bloc("blocs_clash") == []
    # Both at/above the floor -> clash.
    tr.update(0.1, {"red_bloc": 5, "cyan_bloc": 4})
    clash = bus.bloc("blocs_clash")
    assert len(clash) == 1
    assert {clash[0]["a_id"], clash[0]["b_id"]} == {"red_bloc", "cyan_bloc"}
    # Names came from the specs (announcer templates on them).
    assert clash[0]["a_name"] in ("Populist Front", "Civic Union")
    # Held: does not re-fire while both stay violent.
    tr.update(0.1, {"red_bloc": 6, "cyan_bloc": 6})
    assert len(bus.bloc("blocs_clash")) == 1


# ---------------------------------------------------------------------------
# bloc_kettled
# ---------------------------------------------------------------------------


def test_kettled_fires_on_target_and_names_the_other():
    bus, tr = _tracker()
    tr.update(0.1, {"red_bloc": 6, "cyan_bloc": 5}, kettled_faction="red_bloc")
    k = bus.bloc("bloc_kettled")
    assert len(k) == 1
    assert k[0]["a_id"] == "red_bloc"   # the kettled bloc
    assert k[0]["b_id"] == "cyan_bloc"  # the one left to scatter
    # Held while the same bloc stays kettled.
    tr.update(0.1, {"red_bloc": 4, "cyan_bloc": 5}, kettled_faction="red_bloc")
    assert len(bus.bloc("bloc_kettled")) == 1


def test_kettled_rearms_when_target_changes():
    bus, tr = _tracker()
    tr.update(0.1, {"red_bloc": 6, "cyan_bloc": 5}, kettled_faction="red_bloc")
    tr.update(0.1, {"red_bloc": 6, "cyan_bloc": 5}, kettled_faction=None)
    tr.update(0.1, {"red_bloc": 6, "cyan_bloc": 5}, kettled_faction="cyan_bloc")
    k = bus.bloc("bloc_kettled")
    assert [e["a_id"] for e in k] == ["red_bloc", "cyan_bloc"]


def test_unknown_kettle_target_is_ignored():
    bus, tr = _tracker()
    tr.update(0.1, {"red_bloc": 6, "cyan_bloc": 5}, kettled_faction="ghost_bloc")
    assert bus.bloc("bloc_kettled") == []


# ---------------------------------------------------------------------------
# bloc_dominant
# ---------------------------------------------------------------------------


def test_dominant_fires_when_one_bloc_overwhelms():
    bus, tr = _tracker(dominant_cooldown=0.0)
    # 8 vs 2: ratio 4.0 and margin 6 -> red dominant.
    tr.update(0.1, {"red_bloc": 8, "cyan_bloc": 2})
    d = bus.bloc("bloc_dominant")
    assert len(d) == 1
    assert d[0]["a_id"] == "red_bloc"
    assert d[0]["a_count"] == 8 and d[0]["b_count"] == 2


def test_dominant_does_not_fire_on_a_narrow_lead():
    bus, tr = _tracker(dominant_cooldown=0.0)
    # 6 vs 5: neither ratio (1.2 < 1.6) nor margin (1 < 3) -> no dominance.
    tr.update(0.1, {"red_bloc": 6, "cyan_bloc": 5})
    assert bus.bloc("bloc_dominant") == []


def test_dominant_refires_only_on_flip():
    bus, tr = _tracker(dominant_cooldown=0.0)
    tr.update(0.1, {"red_bloc": 9, "cyan_bloc": 2})   # red dominant
    tr.update(0.1, {"red_bloc": 10, "cyan_bloc": 3})  # still red -> no re-fire
    tr.update(0.1, {"red_bloc": 2, "cyan_bloc": 9})   # flip -> cyan dominant
    d = bus.bloc("bloc_dominant")
    assert [e["a_id"] for e in d] == ["red_bloc", "cyan_bloc"]


def test_dominant_respects_cooldown():
    bus, tr = _tracker(dominant_cooldown=8.0)
    tr.update(0.1, {"red_bloc": 9, "cyan_bloc": 2})   # red dominant @ t=0.1
    # Flip to cyan almost immediately: inside cooldown -> suppressed.
    tr.update(0.1, {"red_bloc": 2, "cyan_bloc": 9})
    assert [e["a_id"] for e in bus.bloc("bloc_dominant")] == ["red_bloc"]
    # After the cooldown elapses, a later flip is allowed to speak.
    for _ in range(100):
        tr.update(0.1, {"red_bloc": 2, "cyan_bloc": 9})
    assert "cyan_bloc" in [e["a_id"] for e in bus.bloc("bloc_dominant")]


def test_reset_clears_all_guards():
    bus, tr = _tracker(dominant_cooldown=0.0)
    tr.update(0.1, {"red_bloc": 8, "cyan_bloc": 4}, kettled_faction="red_bloc")
    assert bus.bloc()  # something fired
    tr.reset()
    bus.events.clear()
    # After reset the same conditions re-announce from scratch.
    tr.update(0.1, {"red_bloc": 8, "cyan_bloc": 4}, kettled_faction="red_bloc")
    assert bus.bloc("blocs_clash")
    assert bus.bloc("bloc_kettled")
