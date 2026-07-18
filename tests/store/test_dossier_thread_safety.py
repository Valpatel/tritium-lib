# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Cross-thread regression tests for DossierStore / DossierManager.

BaseStore shares ONE sqlite3 connection across threads
(``check_same_thread=False``) and its contract is that every touch of
``self._conn`` holds ``self._lock``.  The read paths (``get_dossier``,
``find_by_identifier``, ``search``, ``get_recent``) used to skip the lock:
a reader on the DossierManager event-listener thread interleaving with a
lock-holding writer on another thread corrupted the C-level statement
lifecycle and crashed with ``sqlite3.InterfaceError: bad parameter or
other API misuse`` (SQLITE_MISUSE) — observed in production 2026-07-17 at
dossiers.py get_dossier, reached via find_by_identifier <-
find_or_create_for_target <- _handle_geofence_event <-
_event_listener_loop.

These tests drive that exact path from a NON-owning thread while the
owning thread writes.  On the unfixed code they fail within ~0.1 s; the
hammer windows below leave a wide margin without slowing the suite much.
"""

import threading
import time

import pytest

from tritium_lib.store.dossiers import DossierStore
from tritium_lib.tracking.dossier_manager import DossierManager

_MAC = "AA:BB:CC:DD:EE:FF"
_BLE_TARGET = "ble_aabbccddeeff"
_HAMMER_S = 2.5


def _hammer(workers, duration_s=_HAMMER_S):
    """Run worker callables in parallel threads, collecting exceptions.

    Each worker is called in a tight loop on its own thread until the
    stop event fires (duration elapsed or any worker raised).
    """
    errors: list[BaseException] = []
    stop = threading.Event()

    def loop(fn):
        while not stop.is_set():
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 — record and stop
                errors.append(exc)
                stop.set()
                return

    threads = [threading.Thread(target=loop, args=(fn,), daemon=True)
               for fn in workers]
    deadline = time.monotonic() + duration_s
    for t in threads:
        t.start()
    while time.monotonic() < deadline and not stop.is_set():
        time.sleep(0.05)
    stop.set()
    for t in threads:
        t.join(timeout=10)
    return errors


class TestDossierStoreCrossThread:
    """Store-level: unlocked reads used to race lock-holding writes."""

    @pytest.mark.unit
    def test_find_by_identifier_from_non_owning_thread_during_writes(
        self, tmp_path
    ):
        """find_by_identifier -> get_dossier on a foreign thread must
        survive a concurrent add_signal storm (the production stack)."""
        store = DossierStore(tmp_path / "race.db")
        did = store.create_dossier("racer", identifiers={"mac": _MAC})

        errors = _hammer([
            lambda: store.add_signal(did, "ble", "sighting", {"rssi": -60}),
            lambda: store.find_by_identifier("mac", _MAC),
            lambda: store.find_by_identifier("mac", _MAC),
        ])
        store.close()
        assert not errors, f"cross-thread store access raised: {errors[0]!r}"

    @pytest.mark.unit
    def test_read_surface_from_non_owning_threads_during_writes(
        self, tmp_path
    ):
        """The whole read surface (get_dossier / search / get_recent)
        must be serialized against writers on the shared connection."""
        store = DossierStore(tmp_path / "race2.db")
        did = store.create_dossier("sweep", identifiers={"mac": _MAC})

        errors = _hammer([
            lambda: store.add_signal(did, "yolo", "sighting", {"c": 0.9}),
            lambda: store.get_dossier(did),
            lambda: store.search("sweep"),
            lambda: store.get_recent(limit=10),
        ])
        store.close()
        assert not errors, f"cross-thread store access raised: {errors[0]!r}"


class TestDossierManagerListenerPath:
    """Manager-level: the geofence listener path on a foreign thread."""

    @pytest.mark.unit
    def test_geofence_handler_on_listener_thread_during_writes(
        self, tmp_path
    ):
        """_handle_geofence_event (what the event-listener thread runs)
        must survive concurrent store writes from the owning thread.

        The dossier cache entry is popped each iteration so every pass
        re-drives find_or_create_for_target -> find_by_identifier ->
        get_dossier against the store, exactly like a fresh target on
        the real listener thread.

        Honest scope note: this is the full production stack driven
        end-to-end, but the manager's own lock rendezvous points
        phase-lock the threads enough that the race rarely fires within
        the hammer budget — on the unfixed code this test usually still
        passed.  The reliable red/green detectors for the underlying
        store race are the TestDossierStoreCrossThread tests above,
        which hold the reader threads inside the unlocked SELECTs
        continuously (and reproduced the crash in ~0.1 s).  This test
        earns its keep as integration coverage: if the store fix ever
        regresses in a way that breaks the real listener call chain,
        the store-level tests catch the race and this one proves the
        chain itself still completes without error.
        """
        store = DossierStore(tmp_path / "race3.db")
        mgr = DossierManager(store=store)
        # Pre-existing dossier keyed by the MAC the ble_ target derives,
        # so the listener path takes the find_by_identifier branch.
        did = store.create_dossier(
            "phone", identifiers={"mac": _MAC}
        )

        def listener_pass():
            with mgr._lock:
                mgr._target_dossier_map.pop(_BLE_TARGET, None)
            mgr._handle_geofence_event("geofence:enter", {
                "target_id": _BLE_TARGET,
                "zone_name": "perimeter",
                "zone_type": "restricted",
                "zone_id": "z1",
                "position": [1.0, 2.0],
                "timestamp": time.time(),
            })

        errors = _hammer([
            lambda: store.add_signal(did, "ble", "presence", {"rssi": -55}),
            listener_pass,
            listener_pass,
            listener_pass,
        ])
        store.close()
        assert not errors, f"listener-path geofence handling raised: {errors[0]!r}"
