# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Event bus interface — thread-safe and async pub/sub for internal events.

Both tritium-sc and tritium-edge use the same event bus pattern.
This is the shared interface; each project can extend with custom events.
"""

from .bus import EventBus, AsyncEventBus, Event

__all__ = ["EventBus", "AsyncEventBus", "Event"]
