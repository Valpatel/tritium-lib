# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""AddonContext — dependency injection container for addon register().

The addon loader builds an AddonContext from SC internals so addons
never need to fish through app.state for dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .protocols import IEventBus, IMQTTClient, IRouterHandler, ITargetTracker


@dataclass
class AddonContext:
    """Dependency injection context passed to addons during register().

    The addon loader builds this from SC internals.  Addons never need
    to fish through ``app.state`` — everything they need is here.
    """

    # Core services (may be None if not available)
    target_tracker: Optional[ITargetTracker] = None
    event_bus: Optional[IEventBus] = None
    mqtt_client: Optional[IMQTTClient] = None
    router_handler: Optional[IRouterHandler] = None

    # Configuration
    site_id: str = "home"
    data_dir: str = "data"  # writable directory for addon data

    # App state for hot-reload persistence (dict-like)
    state: dict = field(default_factory=dict)

    # Addon event bus (inter-addon communication)
    addon_event_bus: Optional[Any] = None  # AddonEventBus instance

    # Raw app reference (escape hatch — avoid using this)
    app: Optional[Any] = None

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a value from persistent state (survives hot-reload)."""
        return self.state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """Store a value in persistent state."""
        self.state[key] = value

    def has_service(self, name: str) -> bool:
        """Check if a named service attribute is available (non-None)."""
        return getattr(self, name, None) is not None
