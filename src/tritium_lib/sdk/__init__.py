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

__all__ = [
    "AddonBase",
    "AddonInfo",
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
]

SDK_VERSION = "1.0.0"
