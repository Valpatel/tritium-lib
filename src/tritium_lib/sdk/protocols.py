# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""Protocol interfaces for Tritium SDK dependency injection.

These are runtime-checkable Protocol classes (structural typing, not ABCs).
Addons program against these interfaces rather than concrete implementations.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class ITargetTracker(Protocol):
    """Protocol for target tracking systems."""

    def update_target(self, target_id: str, data: dict) -> None: ...
    def get_target(self, target_id: str) -> dict | None: ...
    def get_all_targets(self) -> list[dict]: ...
    def remove_target(self, target_id: str) -> bool: ...


@runtime_checkable
class IEventBus(Protocol):
    """Protocol for event pub/sub systems."""

    def publish(self, topic: str, data: Any = None, source: str = "") -> Any: ...
    def subscribe(self, topic: str, callback: Callable) -> Any: ...


@runtime_checkable
class IMQTTClient(Protocol):
    """Protocol for MQTT clients that support publish and subscribe."""

    def publish(self, topic: str, payload: Any, **kwargs) -> None: ...
    def subscribe(self, topic: str, callback: Callable | None = None) -> None: ...


@runtime_checkable
class IRouterHandler(Protocol):
    """Protocol for registering API routes."""

    def include_router(
        self,
        router: Any,
        prefix: str = "",
        tags: list[str] | None = None,
    ) -> None: ...


@runtime_checkable
class ICommander(Protocol):
    """Protocol for commander implementations.

    Amy is one; others can be swapped in.  Any commander must provide
    status inspection, asset dispatch, narration, and situational awareness.
    """

    def get_status(self) -> dict: ...
    def dispatch(self, target_id: str, waypoints: list) -> bool: ...
    def narrate(self, message: str) -> None: ...
    def get_situation(self) -> dict: ...
