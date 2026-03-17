# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""Addon type interfaces — one per plugin type in the Telegraf-inspired pipeline.

Sensor → Processor → Aggregator → Commander → Bridge
                                                ↓
                                              Output

Plus: Panel (UI), DataSource (external feed), Tool (dev/ops).
"""

from __future__ import annotations

from typing import Any

from .addon_base import AddonBase


class SensorAddon(AddonBase):
    """Collects data from hardware or APIs and produces targets.

    Example: BLE scanner, camera feed, SDR receiver, weather station.

    Override gather() to collect and emit targets.
    """

    async def gather(self) -> list[dict]:
        """Collect data and return a list of target dicts.

        Each dict should have at minimum:
        - target_id: str (unique identifier)
        - source: str (sensor type, e.g., 'ble', 'camera', 'mesh')
        - position: dict with lat/lng or x/y

        Returns:
            List of target dicts to feed into the target tracker.
        """
        return []


class ProcessorAddon(AddonBase):
    """Transforms or enriches individual targets.

    Example: BLE classifier, threat scorer, device fingerprinter.

    Override process() to enrich a target.
    """

    async def process(self, target: dict) -> dict:
        """Enrich or transform a single target.

        Args:
            target: Target dict from the tracker.

        Returns:
            Modified target dict (same ID, added/changed fields).
        """
        return target


class AggregatorAddon(AddonBase):
    """Combines multiple data streams into higher-level insights.

    Example: Target correlator, convoy detector, trilateration.

    Override aggregate() to combine targets.
    """

    async def aggregate(self, targets: list[dict]) -> list[dict]:
        """Combine multiple targets into aggregated results.

        Args:
            targets: List of target dicts.

        Returns:
            List of aggregated/fused target dicts.
        """
        return targets


class CommanderAddon(AddonBase):
    """Makes decisions and dispatches assets.

    Example: Amy AI commander, rule engine, Lua scripting.

    Override think() to evaluate the situation and decide actions.
    """

    async def think(self, situation: dict) -> list[dict]:
        """Evaluate the tactical situation and decide on actions.

        Args:
            situation: Dict containing targets, alerts, zones, resources.

        Returns:
            List of action dicts (dispatch, alert, investigate, etc.).
        """
        return []

    async def speak(self, message: str) -> None:
        """Output a message (narration, alert, commentary).

        Args:
            message: Text to announce/display.
        """
        pass


class BridgeAddon(AddonBase):
    """Connects to external systems for data exchange.

    Example: TAK bridge, MQTT export, webhook, federation.

    Override send() to push data out, and optionally receive() to pull data in.
    """

    async def send(self, targets: list[dict]) -> None:
        """Send target data to the external system.

        Args:
            targets: List of target dicts to export.
        """
        pass

    async def receive(self) -> list[dict]:
        """Receive data from the external system.

        Returns:
            List of target dicts to import.
        """
        return []


class DataSourceAddon(AddonBase):
    """Connects to external data APIs and feeds data into the system.

    Example: Planet Labs satellite, ADS-B Exchange, NOAA weather.

    Override fetch() and process_data() for the sync loop.
    """

    refresh_interval: float = 60.0  # seconds between syncs

    async def fetch(self) -> Any:
        """Fetch data from external source.

        Returns:
            Raw data from the API (format depends on source).
        """
        return None

    async def process_data(self, data: Any) -> list[dict]:
        """Transform raw data into targets/layers/alerts.

        Args:
            data: Raw data from fetch().

        Returns:
            List of target dicts or layer data dicts.
        """
        return []


class PanelAddon(AddonBase):
    """Provides UI panels without backend logic.

    Example: Custom dashboard, visualization widget, report viewer.

    Override get_panels() to define the panels.
    Frontend-only addons don't need register/unregister.
    """
    pass


class ToolAddon(AddonBase):
    """Development or operations tool.

    Example: Test runner, performance monitor, demo mode.

    Override register() to add tool-specific routes and functionality.
    """
    pass
