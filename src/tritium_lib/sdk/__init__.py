# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
#
# NOTE: This SDK module is Apache-2.0 licensed (not AGPL-3.0).
# Private/proprietary addons may import from tritium_lib.sdk freely.
# Do NOT import from other tritium_lib packages in this module.
"""Tritium Addon SDK — interfaces for building addons.

Usage:
    from tritium_lib.sdk import AddonBase, SensorAddon, CommanderAddon
    from tritium_lib.sdk import AddonManifest, load_manifest
"""

from .addon_base import AddonBase, AddonInfo
from .config_loader import AddonConfig
from .context import AddonContext
from .geo_layer import AddonGeoLayer
from .addon_events import AddonEvent, AddonEventBus
from .protocols import IEventBus, IMQTTClient, IRouterHandler, ITargetTracker
from .interfaces import (
    SensorAddon,
    ProcessorAddon,
    AggregatorAddon,
    CommanderAddon,
    BridgeAddon,
    DataSourceAddon,
    PanelAddon,
    ToolAddon,
)
from .manifest import AddonManifest, load_manifest, validate_manifest
from .device_transport import DeviceTransport, LocalTransport, MQTTTransport
from .subprocess_manager import ManagedProcess, SubprocessManager
from .device_registry import DeviceRegistry, RegisteredDevice, DeviceState
from .async_store import AsyncBaseStore
from .runner_base import BaseRunner
from .runner_mqtt import RunnerMQTTClient

__all__ = [
    "AddonBase",
    "AddonConfig",
    "AddonContext",
    "AddonEvent",
    "AddonEventBus",
    "AddonGeoLayer",
    "AddonInfo",
    "IEventBus",
    "IMQTTClient",
    "IRouterHandler",
    "ITargetTracker",
    "SensorAddon",
    "ProcessorAddon",
    "AggregatorAddon",
    "CommanderAddon",
    "BridgeAddon",
    "DataSourceAddon",
    "PanelAddon",
    "ToolAddon",
    "AddonManifest",
    "load_manifest",
    "validate_manifest",
    "DeviceTransport",
    "LocalTransport",
    "MQTTTransport",
    "ManagedProcess",
    "SubprocessManager",
    "DeviceRegistry",
    "RegisteredDevice",
    "DeviceState",
    "AsyncBaseStore",
    "BaseRunner",
    "RunnerMQTTClient",
]

SDK_VERSION = "1.0.0"
