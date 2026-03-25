# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sdk.protocols — runtime-checkable Protocol interfaces."""

from tritium_lib.sdk.protocols import (
    ICommander,
    IEventBus,
    IMQTTClient,
    IRouterHandler,
    ITargetTracker,
)


class TestITargetTracker:
    def test_conforming_class_passes_isinstance(self):
        class MyTracker:
            def update_target(self, target_id: str, data: dict) -> None:
                pass

            def get_target(self, target_id: str) -> dict | None:
                return None

            def get_all_targets(self) -> list[dict]:
                return []

            def remove_target(self, target_id: str) -> bool:
                return False

        assert isinstance(MyTracker(), ITargetTracker)

    def test_non_conforming_class_fails(self):
        class NotATracker:
            def update_target(self, target_id: str, data: dict) -> None:
                pass
            # Missing other methods

        assert not isinstance(NotATracker(), ITargetTracker)


class TestIEventBus:
    def test_conforming_class_passes(self):
        class MyBus:
            def publish(self, topic, data=None, source=""):
                pass

            def subscribe(self, topic, callback):
                pass

        assert isinstance(MyBus(), IEventBus)

    def test_non_conforming_fails(self):
        class NoBus:
            def emit(self, event):
                pass

        assert not isinstance(NoBus(), IEventBus)


class TestIMQTTClient:
    def test_conforming_class_passes(self):
        class MyMQTT:
            def publish(self, topic, payload, **kwargs):
                pass

            def subscribe(self, topic, callback=None):
                pass

        assert isinstance(MyMQTT(), IMQTTClient)


class TestIRouterHandler:
    def test_conforming_class_passes(self):
        class MyRouter:
            def include_router(self, router, prefix="", tags=None):
                pass

        assert isinstance(MyRouter(), IRouterHandler)


class TestICommander:
    def test_conforming_class_passes(self):
        class MyCommander:
            def get_status(self):
                return {}

            def dispatch(self, target_id, waypoints):
                return True

            def narrate(self, message):
                pass

            def get_situation(self):
                return {}

        assert isinstance(MyCommander(), ICommander)

    def test_non_conforming_fails(self):
        class Partial:
            def get_status(self):
                return {}

        assert not isinstance(Partial(), ICommander)
