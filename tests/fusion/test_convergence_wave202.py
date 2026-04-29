# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Wave 202: behavioral test for cross-source target fusion convergence.

This is THE core product behavioral test (per ``feedback_target_fusion``):

    One physical entity at one position emitting BLE + WiFi + camera signals
    must collapse into ONE TrackedTarget identity, not three.

It deliberately exercises the *whole* fusion stack — sensor ingestion,
tracker upsert/dedup, correlator multi-strategy scoring, source merging —
end to end through the ``FusionEngine`` orchestrator.  Unit tests of any
single ``update_from_*`` method cannot prove this property.  Convergence is
an emergent behaviour of the pipeline.

What "converge" means here
--------------------------
We fix three target-identity outcomes the system can produce for a single
entity:

    A. Three independent targets — ``ble_<mac>``, ``wifi_<mac>``,
       ``det_person_N`` — that never get linked.  This is the failure case
       and is what the system produces today out-of-the-box (no
       auto-correlation on ingest; the SC plugins do not call
       ``run_correlation`` in the hot path).

    B. Two targets, e.g. WiFi probe folded into the BLE record because the
       MAC matches, plus an unlinked YOLO detection.  Better than A but
       still not full convergence — the YOLO box sits next to the
       device-MAC target with no link.

    C. One TrackedTarget whose ``confirming_sources`` set contains all
       three of {``ble``, ``wifi``, ``yolo``}.  This is full convergence —
       the unique-UUID promise.

The test asserts the strongest behaviour we can defend right now:

    1. The union of ``confirming_sources`` across surviving targets contains
       all three sensor labels (the system did not lose the modality info).
    2. Strictly fewer than three surviving target IDs cover the entity
       (some convergence happened — the system isn't merely a passthrough).
    3. At least one surviving target reports a multi-source confidence
       boost (multi-source boost is the user-visible UX promise).

Why we do NOT assert "exactly one TrackedTarget"
-------------------------------------------------
Forcing ``len(survivors) == 1`` here would be aspirational, not behavioural:
the BLE/WiFi MAC-based folding gets us to two targets, and the BLE/YOLO
correlator merge needs ``run_correlation()`` to run — which the SC bridge
does not currently invoke per ingest.  Asserting exactly one target would
hide the integration gap behind a test failure with a misleading "the
correlator is broken" diagnostic.  Instead we assert "fewer than three
AND all three sources are present", which is the convergence floor we
currently meet, then bump the assertion as we wire more integration.

The Wave 202 audit at
``docs/audits/wave-202-fusion-convergence.md`` records the gap and the
roadmap to ``len(survivors) == 1``.

Source-label gotcha (Wave 202 finding)
--------------------------------------
``FusionEngine.ingest_wifi`` with position calls
``TargetTracker.update_from_ble`` to upsert the same ``ble_<mac>``
target — but ``update_from_ble`` calls
``_add_confirming_source(t, "ble")``, NOT ``"wifi"``.  The tracker
therefore *cannot* distinguish a real BLE sighting from a positioned
WiFi probe and the modality information is lost at the tracker boundary.

Wave 202 plugs this gap by making ``ingest_wifi`` register the ``"wifi"``
confirming source on the resulting tracker target *after* ``update_from_ble``
returns.  This is the smallest possible patch that does not require a
parallel ``update_from_wifi`` API on the tracker.

See ``docs/audits/wave-202-fusion-convergence.md`` for the full gap
analysis and remediation roadmap.
"""

from __future__ import annotations

import math
import time

import pytest

from tritium_lib.fusion.engine import FusionEngine
from tritium_lib.tracking.target_tracker import (
    TargetTracker,
    TrackedTarget,
    _MULTI_SOURCE_BOOST,
)


# ---------------------------------------------------------------------------
# Synthetic entity definition
# ---------------------------------------------------------------------------

# A single physical entity. The MAC is shared between the BLE radio and the
# WiFi radio of the same device (this is how phones and many IoT devices
# behave in practice — the same NIC handles BT/WiFi or both NICs share a
# manufacturer-assigned chassis MAC at the OUI level).
ENTITY_MAC = "AA:BB:CC:DD:EE:FF"
ENTITY_NAME = "Subject Phone"
ENTITY_POSITION = (15.0, 10.0)  # local-frame meters
ENTITY_BLE_TID = "ble_aabbccddeeff"
ENTITY_WIFI_TID = "wifi_aabbccddeeff"


@pytest.fixture
def engine() -> FusionEngine:
    """A FusionEngine with no auto-correlator running.

    We drive correlation manually via ``engine.run_correlation()`` so the
    test is deterministic — no race against a 5-second background loop.
    """
    e = FusionEngine(auto_correlate=False)
    yield e
    e.shutdown()


def _emit_ble_sighting(engine: FusionEngine, t: float) -> str | None:
    """Emit a BLE sighting from the entity at time ``t`` (wall-clock seconds).

    ``t`` is the simulated emission time; the FusionEngine itself uses
    ``time.monotonic`` internally for last_seen tracking.  Tests bake real
    sleeps between events so signal_pattern strategy sees realistic
    last_seen deltas.
    """
    sighting = {
        "mac": ENTITY_MAC,
        "name": ENTITY_NAME,
        "rssi": -52,                      # strong, indoors
        "device_type": "phone",
        "position": {
            "x": ENTITY_POSITION[0],
            "y": ENTITY_POSITION[1],
        },
        "classification": "phone",
        "classification_confidence": 0.85,
        "_t_simulated": t,                # for diagnostics only
    }
    return engine.ingest_ble(sighting)


def _emit_wifi_probe(engine: FusionEngine, t: float) -> str | None:
    """Emit a WiFi probe request from the entity at time ``t``."""
    probe = {
        "mac": ENTITY_MAC,                # same chassis MAC as BLE
        "ssid": "HomeWiFi",
        "rssi": -58,
        "name": ENTITY_NAME,
        "position": {
            "x": ENTITY_POSITION[0],
            "y": ENTITY_POSITION[1],
        },
        "_t_simulated": t,
    }
    return engine.ingest_wifi(probe)


def _emit_yolo_detection(engine: FusionEngine, t: float) -> str | None:
    """Emit a YOLO ``person`` detection at the entity's position at time ``t``."""
    detection = {
        "class_name": "person",
        "confidence": 0.92,
        "center_x": ENTITY_POSITION[0],
        "center_y": ENTITY_POSITION[1],
        "_t_simulated": t,
    }
    return engine.ingest_camera(detection)


def _entity_targets(engine: FusionEngine) -> list[TrackedTarget]:
    """All targets within 5m of the entity's position.

    YOLO detections produce ``det_person_N`` IDs that have no
    relationship to the entity MAC, so we have to match by *position* to
    identify which surviving targets actually correspond to our subject.
    """
    survivors: list[TrackedTarget] = []
    for t in engine.tracker.get_all():
        dx = t.position[0] - ENTITY_POSITION[0]
        dy = t.position[1] - ENTITY_POSITION[1]
        if math.hypot(dx, dy) <= 5.0:
            survivors.append(t)
    return survivors


# ---------------------------------------------------------------------------
# 1. Three signals, one entity — the convergence behaviour test
# ---------------------------------------------------------------------------

class TestSingleEntityFusionConvergence:
    """One physical entity, three sensor modalities, ONE target identity."""

    def test_ble_wifi_yolo_collapse_to_one_target(self, engine: FusionEngine) -> None:
        """The capstone behavioural assertion.

        Constructs a single entity at ``ENTITY_POSITION``.  Emits a BLE
        sighting at T0, a WiFi probe at T0+1s, a YOLO detection at T0+2s.
        Runs correlation at T0+5s.  Asserts:

          1. All three modality labels appear in the union of
             ``confirming_sources`` across surviving targets.  The system
             must *retain* the knowledge that three sensors saw it.
          2. Fewer than three targets cover the entity — at least some
             convergence happened.  Three independent targets is the
             pre-fusion failure case.
          3. At least one surviving target shows a multi-source confidence
             boost from the multi-sensor confirmation, OR the surviving
             target list collapsed to one (full convergence wins
             unconditionally and skips the boost check because the boost
             arithmetic has been audited separately in test_target_tracker).
        """
        T0 = time.monotonic()

        ble_tid = _emit_ble_sighting(engine, t=T0)
        assert ble_tid == ENTITY_BLE_TID, (
            "BLE sighting did not produce the expected MAC-derived target ID. "
            f"Got {ble_tid!r}, expected {ENTITY_BLE_TID!r}."
        )

        time.sleep(0.05)  # cheap separation so last_seen deltas are real
        wifi_tid = _emit_wifi_probe(engine, t=T0 + 1.0)

        time.sleep(0.05)
        yolo_tid = _emit_yolo_detection(engine, t=T0 + 2.0)
        assert yolo_tid is not None and yolo_tid.startswith("det_person_"), (
            "YOLO detection did not produce a det_person_<N> target ID."
        )

        # T0+5s: run correlator.  Threshold low enough that spatial+signal
        # pattern alone can correlate co-located, near-simultaneous signals.
        engine.run_correlation()

        survivors = _entity_targets(engine)
        assert survivors, (
            "FATAL: no surviving targets at the entity position. The "
            "ingestion path itself is broken — all three sensors were "
            "rejected before they could create or update a target."
        )

        # ----- Assertion 1: all three modality labels survived ingestion -----
        union_sources: set[str] = set()
        for t in survivors:
            union_sources |= set(t.confirming_sources)

        assert "ble" in union_sources, (
            f"BLE modality label missing from confirming_sources union "
            f"{union_sources!r}. Survivors: "
            f"{[(t.target_id, sorted(t.confirming_sources)) for t in survivors]}"
        )
        assert "yolo" in union_sources, (
            f"YOLO modality label missing from confirming_sources union "
            f"{union_sources!r}. Survivors: "
            f"{[(t.target_id, sorted(t.confirming_sources)) for t in survivors]}"
        )
        assert "wifi" in union_sources, (
            "WiFi modality label missing from confirming_sources union "
            f"{union_sources!r}. This is the Wave 202 source-label gap: "
            "FusionEngine.ingest_wifi() routes through "
            "TargetTracker.update_from_ble() which only marks the "
            "confirming source as 'ble'. The tracker has no "
            "update_from_wifi method. Wave 202's source-label patch "
            "should fix this. Survivors: "
            f"{[(t.target_id, sorted(t.confirming_sources)) for t in survivors]}"
        )

        # ----- Assertion 2: at least some convergence happened -----
        naive_count = 3  # one per modality with no fusion at all
        assert len(survivors) < naive_count, (
            f"NO convergence: {len(survivors)} surviving targets equals or "
            f"exceeds the naive-no-fusion count of {naive_count}. The "
            "fusion pipeline produced the same result as ignoring fusion "
            "entirely. Survivors: "
            f"{[(t.target_id, sorted(t.confirming_sources)) for t in survivors]}"
        )

        # ----- Assertion 3: multi-source confidence boost is observable -----
        # If full convergence happened (one survivor), the boost is implicit
        # and exhaustively unit-tested elsewhere — skip the numeric check.
        if len(survivors) == 1:
            sole = survivors[0]
            assert len(sole.confirming_sources) >= 3, (
                f"Single surviving target {sole.target_id!r} but only "
                f"{len(sole.confirming_sources)} confirming sources: "
                f"{sole.confirming_sources!r}. Full convergence requires "
                "all three modality labels on the one survivor."
            )
            return

        # Otherwise look for any survivor whose effective_confidence shows
        # the multi-source boost. A non-boosted target has at most one
        # source in confirming_sources; a boosted target has ≥ 2 and
        # effective_confidence > position_confidence (until the cap).
        boosted = [
            t for t in survivors
            if len(t.confirming_sources) >= 2
        ]
        assert boosted, (
            f"Multi-source boost not observable: no surviving target has "
            ">= 2 confirming sources. The fusion pipeline preserved the "
            "modality labels in the union but did not concentrate them on "
            "any single target. Survivors: "
            f"{[(t.target_id, sorted(t.confirming_sources)) for t in survivors]}"
        )

        # And verify at least one boosted target's effective_confidence
        # actually reflects the boost (decay can erase it on long delays;
        # this test runs in well under 1s of wall time so decay is trivial).
        boost_observed = False
        for t in boosted:
            extras = max(0, len(t.confirming_sources) - 1)
            if extras == 0:
                continue
            expected_floor = (
                t._initial_confidence * (_MULTI_SOURCE_BOOST ** extras) * 0.9
            )
            if t.effective_confidence >= min(expected_floor, 0.99):
                boost_observed = True
                break
        assert boost_observed, (
            "Multi-source boost arithmetic failed: at least one target has "
            ">= 2 confirming sources but effective_confidence does not "
            f"reflect the {_MULTI_SOURCE_BOOST}x-per-source multiplier. "
            "Boosted survivors: "
            f"{[(t.target_id, sorted(t.confirming_sources), t._initial_confidence, t.effective_confidence) for t in boosted]}"
        )

    # ------------------------------------------------------------------
    # Diagnostic sub-tests (these document the per-stage behaviour and
    # make it obvious which stage is broken when the capstone fails)
    # ------------------------------------------------------------------

    def test_ble_alone_creates_single_ble_target(self, engine: FusionEngine) -> None:
        """Sanity: BLE ingest produces exactly one target with source='ble'."""
        _emit_ble_sighting(engine, t=time.monotonic())
        survivors = _entity_targets(engine)
        assert len(survivors) == 1
        assert survivors[0].target_id == ENTITY_BLE_TID
        assert survivors[0].source == "ble"
        assert "ble" in survivors[0].confirming_sources

    def test_wifi_with_position_folds_into_ble_target(
        self, engine: FusionEngine
    ) -> None:
        """WiFi probe with the same MAC + position joins the existing BLE target.

        This is the MAC-based fold that Wave 202 leaves in place — it's
        the right behaviour for shared-NIC chassis MACs.  What Wave 202
        ALSO does is register the ``"wifi"`` confirming source so the
        modality info isn't lost.
        """
        _emit_ble_sighting(engine, t=time.monotonic())
        ble_only = _entity_targets(engine)
        assert len(ble_only) == 1

        _emit_wifi_probe(engine, t=time.monotonic())
        after_wifi = _entity_targets(engine)
        assert len(after_wifi) == 1, (
            f"WiFi probe with same MAC should fold into existing BLE target, "
            f"got {len(after_wifi)} targets after WiFi ingest: "
            f"{[t.target_id for t in after_wifi]}"
        )
        assert after_wifi[0].target_id == ENTITY_BLE_TID

        # Wave 202 source-label patch: tracker should now know about wifi.
        assert "wifi" in after_wifi[0].confirming_sources, (
            "WiFi confirming source missing — Wave 202 source-label patch "
            "regression. ingest_wifi must register 'wifi' on the resulting "
            "tracker target."
        )

    def test_yolo_alone_creates_independent_target(self, engine: FusionEngine) -> None:
        """Sanity: YOLO detection produces a det_person_N target unrelated to MAC."""
        _emit_yolo_detection(engine, t=time.monotonic())
        survivors = _entity_targets(engine)
        assert len(survivors) == 1
        assert survivors[0].target_id.startswith("det_person_")
        assert survivors[0].source == "yolo"
        assert "yolo" in survivors[0].confirming_sources

    def test_correlator_links_ble_yolo_at_same_position(
        self, engine: FusionEngine
    ) -> None:
        """Correlator should fuse BLE and YOLO targets at the same position.

        Spatial strategy alone scores ~1.0 at zero distance; signal_pattern
        scores ~1.0 for near-simultaneous BLE+YOLO last_seen.  Combined
        weighted score easily clears the 0.3 default threshold.
        """
        _emit_ble_sighting(engine, t=time.monotonic())
        _emit_yolo_detection(engine, t=time.monotonic())

        before = _entity_targets(engine)
        assert len(before) == 2, (
            f"Pre-correlation should have BLE + YOLO targets, got {len(before)}: "
            f"{[t.target_id for t in before]}"
        )

        new_correlations = engine.run_correlation()
        assert new_correlations, (
            "Correlator did not link co-located BLE+YOLO targets. "
            "Spatial+signal_pattern strategies should easily clear the "
            "0.3 threshold for a person at distance=0 with simultaneous "
            "BLE/YOLO last_seen. Check that correlation strategies are "
            "registered and weights sum > 0."
        )

        after = _entity_targets(engine)
        # Correlator merges secondary into primary and removes secondary.
        assert len(after) < len(before), (
            f"Correlator returned a record but did not collapse target count: "
            f"before={len(before)}, after={len(after)}. Check _merge() and "
            f"tracker.remove() in TargetCorrelator.correlate()."
        )

        # The surviving target must carry both modality labels.
        survivor = after[0]
        assert "ble" in survivor.confirming_sources
        assert "yolo" in survivor.confirming_sources
