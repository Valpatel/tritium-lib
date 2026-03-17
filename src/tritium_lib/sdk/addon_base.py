# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""Base class for all Tritium addons.

Every addon subclasses AddonBase and implements register/unregister.
The addon loader calls these during enable/disable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .addon_events import AddonEvent, AddonEventBus


@dataclass
class AddonInfo:
    """Static metadata about an addon."""
    id: str
    name: str
    version: str = "0.0.0"
    description: str = ""
    author: str = ""
    license: str = "AGPL-3.0"
    category: str = "system"  # Which consolidated window to join
    icon: str = ""
    min_sdk_version: str = "1.0.0"


class AddonBase:
    """Base class for all Tritium addons.

    Subclass this and implement register() and unregister().

    Example::

        class MyAddon(AddonBase):
            info = AddonInfo(id="my-addon", name="My Addon", version="1.0.0")

            async def register(self, app):
                # Wire up routes, start services, subscribe to events
                pass

            async def unregister(self, app):
                # Clean shutdown, close connections, unsubscribe
                pass
    """

    info: AddonInfo = AddonInfo(id="unknown", name="Unknown")

    def __init__(self):
        self._registered = False
        self._background_tasks: list = []
        self._mqtt_subscriptions: list = []
        self._event_subscriptions: list = []
        self._addon_event_bus: AddonEventBus | None = None
        self._addon_event_unsubs: list[Callable] = []

    async def register(self, app: Any) -> None:
        """Called when the addon is enabled.

        Override this to:
        - Add FastAPI routes (app.include_router)
        - Start background tasks
        - Subscribe to MQTT topics
        - Subscribe to event bus events
        - Initialize hardware connections

        Args:
            app: The Tritium application context. Provides:
                - app.event_bus: EventBus for pub/sub
                - app.mqtt: MQTT client (if available)
                - app.database_url: Database connection string
                - app.include_router: Add FastAPI routes
                - app.config: Application configuration
        """
        self._registered = True

    async def unregister(self, app: Any) -> None:
        """Called when the addon is disabled.

        Override this to clean up everything register() set up.
        MUST be safe to call even if register() partially failed.
        MUST complete within 10 seconds.

        Args:
            app: Same application context as register().
        """
        # Cancel background tasks
        for task in self._background_tasks:
            if hasattr(task, 'cancel'):
                task.cancel()
        self._background_tasks.clear()

        # Unsubscribe MQTT
        for unsub in self._mqtt_subscriptions:
            if callable(unsub):
                unsub()
        self._mqtt_subscriptions.clear()

        # Unsubscribe events
        for unsub in self._event_subscriptions:
            if callable(unsub):
                unsub()
        self._event_subscriptions.clear()

        # Unsubscribe addon events
        for unsub in self._addon_event_unsubs:
            unsub()
        self._addon_event_unsubs.clear()

        self._registered = False

    # ------------------------------------------------------------------
    # Inter-addon event helpers
    # ------------------------------------------------------------------

    def set_event_bus(self, bus: AddonEventBus) -> None:
        """Attach an :class:`AddonEventBus` to this addon."""
        self._addon_event_bus = bus

    def publish_addon_event(
        self,
        event_type: str,
        data: dict,
        device_id: str = "",
    ) -> AddonEvent | None:
        """Publish an addon event using ``self.info.id`` as the source.

        Returns the created :class:`AddonEvent`, or ``None`` if no bus is set.
        """
        if self._addon_event_bus is None:
            return None
        return self._addon_event_bus.publish(
            source_addon=self.info.id,
            event_type=event_type,
            data=data,
            device_id=device_id,
        )

    def subscribe_addon_event(
        self,
        pattern: str,
        callback: Callable[[AddonEvent], None],
    ) -> None:
        """Subscribe to addon events and track for cleanup on unregister."""
        if self._addon_event_bus is None:
            return
        unsub = self._addon_event_bus.subscribe(pattern, callback)
        self._addon_event_unsubs.append(unsub)

    def get_panels(self) -> list[dict]:
        """Return panel definitions for the frontend.

        Override to provide panels. Each panel dict has:
        - id: unique panel ID
        - title: display title
        - file: JS module path (relative to addon's frontend/ dir)
        - category: which consolidated window (default: self.info.category)
        - tab_order: position within window (default: 99)

        Returns:
            List of panel definition dicts.
        """
        return []

    def get_layers(self) -> list[dict]:
        """Return map layer definitions.

        Override to provide layers. Each layer dict has:
        - id: unique layer ID
        - label: display label
        - category: which layer category
        - color: hex color for swatch
        - key: state key for toggle (default: 'show' + capitalized id)

        Returns:
            List of layer definition dicts.
        """
        return []

    def get_context_menu_items(self) -> list[dict]:
        """Return context menu items for the map right-click menu.

        Each item has:
        - label: display text
        - action: event name to emit
        - when: condition expression (e.g., "target.source == 'mesh'")

        Returns:
            List of context menu item dicts.
        """
        return []

    def get_shortcuts(self) -> list[dict]:
        """Return keyboard shortcut bindings.

        Each binding has:
        - key: key combo (e.g., "Shift+M")
        - action: event name to emit
        - description: human-readable description

        Returns:
            List of shortcut binding dicts.
        """
        return []

    def health_check(self) -> dict:
        """Return addon health status.

        Override to provide addon-specific health checks.

        Returns:
            Dict with 'status' ('ok', 'degraded', 'error') and optional 'detail'.
        """
        return {"status": "ok" if self._registered else "not_registered"}

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.info.id} v={self.info.version}>"
