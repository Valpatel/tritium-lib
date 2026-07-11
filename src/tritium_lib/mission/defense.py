# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Key-terrain defender placement — garrison the bridges that matter.

Pure, deterministic tactical helper: given the **chokepoint tactical objects**
the GIS layer already projects (``hold_value`` / ``sever`` / ``key_terrain`` —
see :func:`tritium_lib.geo.gis.chokepoints.chokepoint_tactical_object`) and a
set of defender units, decide which crossings to hold and which unit holds each.

Both halves of the mantra:

* **Fun / gamified:** a defender force auto-garrisons the *tactically real*
  bridges — the primary-road river spans a rival must funnel through — instead
  of spreading thin across every ditch culvert.  The player sees units move to
  the crossings that decide the fight.
* **Production:** the same key-terrain-aware placement a real security
  deployment wants — station scarce assets on the severable links that deny the
  most movement, nearest unit first to minimise repositioning.

Lane-agnostic by construction: it consumes plain dicts (the open chokepoint
contract) and defender dicts, imports only the shared geo distance helper, and
returns plain assignment dicts.  It does **not** import or touch the GIS
providers — the combat / riot / costmap lanes feed it whatever chokepoints they
already have.  No IO, no RNG, no engine coupling.
"""

from __future__ import annotations

from typing import Any, Iterable

from tritium_lib.geo import approx_distance_m

__all__ = ["rank_hold_points", "assign_defenders_to_chokepoints"]


def _lonlat(obj: Any) -> tuple[float, float] | None:
    """Extract ``(lon, lat)`` from a flexible position-bearing dict.

    Accepts ``{"position": {"lon":.., "lat":..}}``, ``{"lon":.., "lat":..}``,
    ``{"position": [lon, lat]}`` / ``(lon, lat)``.  Returns ``None`` when no
    usable coordinate is present (the object is skipped rather than raising).
    """
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        try:
            return float(obj[0]), float(obj[1])
        except (TypeError, ValueError):
            return None
    if not isinstance(obj, dict):
        return None
    pos = obj.get("position", obj)
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        try:
            return float(pos[0]), float(pos[1])
        except (TypeError, ValueError):
            return None
    if isinstance(pos, dict):
        lon = pos.get("lon", pos.get("lng", pos.get("x")))
        lat = pos.get("lat", pos.get("y"))
        if lon is not None and lat is not None:
            try:
                return float(lon), float(lat)
            except (TypeError, ValueError):
                return None
    return None


def _hold_value(cp: dict) -> int:
    try:
        return int(cp.get("hold_value", 0))
    except (TypeError, ValueError):
        return 0


def _is_key_terrain(cp: dict) -> bool:
    return "key_terrain" in (cp.get("tags") or []) or _hold_value(cp) >= 6


def rank_hold_points(
    chokepoints: Iterable[dict], *, min_hold: int = 1
) -> list[dict]:
    """Order chokepoint tactical objects best-defensive-first.

    Sort key (all descending in priority, deterministic): ``hold_value``, then
    key-terrain, then severable (blowing it denies a route), then a stable
    ``id`` tie-break so the ordering never depends on input noise.  Chokepoints
    with ``hold_value < min_hold`` (or no usable position) are dropped.
    """
    usable = [
        cp for cp in chokepoints
        if isinstance(cp, dict)
        and _lonlat(cp) is not None
        and _hold_value(cp) >= min_hold
    ]
    return sorted(
        usable,
        key=lambda cp: (
            -_hold_value(cp),
            0 if _is_key_terrain(cp) else 1,
            0 if cp.get("sever") else 1,
            str(cp.get("id", "")),
        ),
    )


def assign_defenders_to_chokepoints(
    chokepoints: Iterable[dict],
    defenders: Iterable[dict],
    *,
    max_per_point: int = 1,
    min_hold: int = 1,
) -> list[dict]:
    """Greedily garrison the highest-value crossings, nearest unit first.

    For each hold-point (highest ``hold_value`` first, up to ``max_per_point``
    defenders each) assign the closest not-yet-assigned defender.  This places
    scarce defenders on the crossings that decide the fight while minimising how
    far each unit must move.

    Args:
        chokepoints: iterable of chokepoint tactical objects (the open contract
            with ``id`` / ``position`` / ``hold_value`` / ``sever`` / ``tags``).
        defenders: iterable of defender dicts, each with an ``id`` and a
            position (``{"position": {"lon","lat"}}`` or ``{"lon","lat"}`` etc.).
        max_per_point: how many defenders may garrison one crossing (>=1).
        min_hold: ignore crossings below this ``hold_value``.

    Returns:
        A list of assignment dicts, one per assigned defender, in descending
        hold-value order::

            {defender_id, chokepoint_id, hold_value, kind,
             position: {lon, lat}, distance_m, key_terrain, sever}

        Empty when there are no defenders or no qualifying chokepoints.
        Deterministic — no RNG; ties break on id.
    """
    ranked = rank_hold_points(chokepoints, min_hold=min_hold)
    slots = max(1, int(max_per_point))

    # (id, lon, lat) for every defender with a usable position; kept as a
    # mutable pool so each unit is assigned at most once.
    pool: list[tuple[str, float, float]] = []
    for d in defenders:
        if not isinstance(d, dict):
            continue
        ll = _lonlat(d)
        if ll is None:
            continue
        pool.append((str(d.get("id", d.get("target_id", ""))), ll[0], ll[1]))

    # Fill each hold-point to ``slots`` capacity in ranked order: the most
    # important crossing is manned (up to the cap) before any defender is spent
    # on a lesser one — concentrating scarce assets on key terrain.  With the
    # default cap of 1 this is one-defender-per-point down the priority list.
    assignments: list[dict] = []
    for cp in ranked:
        if not pool:
            break
        clon, clat = _lonlat(cp)  # ranked -> always present
        for _slot in range(slots):
            if not pool:
                break
            # Nearest defender (deterministic id tie-break for equal ranges).
            best_i = min(
                range(len(pool)),
                key=lambda i: (
                    approx_distance_m(clat, clon, pool[i][2], pool[i][1]),
                    pool[i][0],
                ),
            )
            did, dlon, dlat = pool.pop(best_i)
            assignments.append({
                "defender_id": did,
                "chokepoint_id": cp.get("id"),
                "hold_value": _hold_value(cp),
                "kind": cp.get("kind", "bridge"),
                "position": {"lon": clon, "lat": clat},
                "distance_m": round(approx_distance_m(clat, clon, dlat, dlon), 1),
                "key_terrain": _is_key_terrain(cp),
                "sever": bool(cp.get("sever")),
            })
    return assignments
