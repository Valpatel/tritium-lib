# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Shared persistence stores for the Tritium ecosystem."""

from .base import BaseStore
from .ble import BleStore
from .config_store import ConfigStore
from .dossiers import DossierStore
from .reid import ReIDStore
from .targets import TargetStore
from .audit_log import AuditStore, AuditEntry, AuditSeverity
from .event_store import EventStore, TacticalEvent, SEVERITY_LEVELS
from .screenshot_store import ScreenshotStore
from .embodiment_registry import (
    EmbodimentRegistry,
    REGISTRY as EMBODIMENT_REGISTRY,
    register_embodiment,
    is_occupied,
    pop_pending_action,
    occupied_ids,
    set_perception,
    get_perception,
    deregister_embodiment_silent,
    record_graphling_kill,
    record_graphling_score,
    configure_persistence as configure_embodiment_persistence,
)

__all__ = [
    "BaseStore",
    "BleStore",
    "ConfigStore",
    "DossierStore",
    "EventStore",
    "ReIDStore",
    "ScreenshotStore",
    "TargetStore",
    "TacticalEvent",
    "SEVERITY_LEVELS",
    "AuditStore",
    "AuditEntry",
    "AuditSeverity",
    "EmbodimentRegistry",
    "EMBODIMENT_REGISTRY",
    "register_embodiment",
    "is_occupied",
    "pop_pending_action",
    "occupied_ids",
    "set_perception",
    "get_perception",
    "deregister_embodiment_silent",
    "record_graphling_kill",
    "record_graphling_score",
    "configure_embodiment_persistence",
]
