# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""Standalone headless runner for the Weather Station addon.

Run on a Raspberry Pi or any Linux box to collect weather data and
publish it to a Tritium Command Center via MQTT — no browser needed.

Usage::

    python -m tritium_lib.sdk.examples.weather_station.runner
    python -m tritium_lib.sdk.examples.weather_station.runner --station wx-roof-01 --interval 30
    python -m tritium_lib.sdk.examples.weather_station.runner --mqtt-host 192.168.1.100

This is the reference implementation for ``BaseRunner``-based addon runners.
Copy and adapt for your own addon.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import Any

from ...runner_base import BaseRunner
from . import WeatherSimulator

logger = logging.getLogger(__name__)


class WeatherStationRunner(BaseRunner):
    """Headless runner that publishes simulated weather over MQTT.

    Extends ``BaseRunner`` to demonstrate the full runner lifecycle:
    discover_devices, start_device, stop_device, on_command.
    """

    def __init__(
        self,
        station_id: str = "wx-sim-001",
        poll_interval: float = 10.0,
        lat: float = 39.7392,
        lng: float = -104.9903,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=station_id,
            device_type="weather",
            **kwargs,
        )
        self.station_id = station_id
        self.poll_interval = poll_interval
        self.lat = lat
        self.lng = lng
        self._simulator: WeatherSimulator | None = None
        self._poll_task: asyncio.Task | None = None
        self._reading_count: int = 0

    # ------------------------------------------------------------------
    # BaseRunner abstract methods
    # ------------------------------------------------------------------

    async def discover_devices(self) -> list[dict[str, Any]]:
        """Return a single simulated weather station 'device'."""
        return [
            {
                "id": self.station_id,
                "type": "weather_station_sim",
                "lat": self.lat,
                "lng": self.lng,
            }
        ]

    async def start_device(self, device_info: dict[str, Any]) -> bool:
        """Create the simulator and start the polling loop."""
        self._simulator = WeatherSimulator(
            station_id=device_info["id"],
            lat=device_info.get("lat", self.lat),
            lng=device_info.get("lng", self.lng),
        )
        self._poll_task = asyncio.ensure_future(self._poll_loop())
        logger.info(
            "Started weather station %s (poll every %.1fs)",
            device_info["id"],
            self.poll_interval,
        )
        return True

    async def stop_device(self, device_id: str) -> bool:
        """Cancel the polling task."""
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
            self._poll_task = None
        self._simulator = None
        logger.info("Stopped weather station %s", device_id)
        return True

    async def on_command(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle remote commands.

        Supported commands:
        - ``get_reading``: return the latest weather reading immediately
        - ``set_interval``: change the polling interval
        - ``get_status``: return runner status
        """
        if command == "get_reading":
            if self._simulator is None:
                return {"error": "not_started"}
            reading = self._simulator.read()
            return reading.to_dict()

        if command == "set_interval":
            new_interval = payload.get("interval", self.poll_interval)
            self.poll_interval = float(new_interval)
            return {"interval": self.poll_interval}

        if command == "get_status":
            return {
                "station_id": self.station_id,
                "reading_count": self._reading_count,
                "poll_interval": self.poll_interval,
                "running": self._poll_task is not None and not self._poll_task.done(),
            }

        return {"error": f"unknown_command: {command}"}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Periodically read weather data and publish over MQTT."""
        while True:
            try:
                if self._simulator is not None:
                    reading = self._simulator.read()
                    self._reading_count += 1

                    # Publish the reading
                    self._publish_data(
                        data_type="reading",
                        data=reading.to_dict(),
                        device_id=self.station_id,
                    )

                    # Publish alert if severity > 1
                    if reading.severity_level() >= 2:
                        self._publish_data(
                            data_type="alert",
                            data={
                                "station_id": self.station_id,
                                "conditions": reading.classify_conditions(),
                                "severity": reading.severity_level(),
                                "reading": reading.to_dict(),
                            },
                            device_id=self.station_id,
                        )

                    if self._reading_count % 10 == 0:
                        logger.info(
                            "Station %s: %d readings, latest=%.1fC %s",
                            self.station_id,
                            self._reading_count,
                            reading.temperature_c,
                            reading.classify_conditions(),
                        )
            except Exception:
                logger.exception("Error in weather poll loop")

            await asyncio.sleep(self.poll_interval)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the weather station runner."""
    parser = argparse.ArgumentParser(
        description="Tritium Weather Station Runner — headless MQTT weather sensor",
    )
    parser.add_argument(
        "--station", default="wx-sim-001", help="Station ID (default: wx-sim-001)"
    )
    parser.add_argument(
        "--interval", type=float, default=10.0, help="Poll interval in seconds (default: 10)"
    )
    parser.add_argument(
        "--lat", type=float, default=39.7392, help="Station latitude"
    )
    parser.add_argument(
        "--lng", type=float, default=-104.9903, help="Station longitude"
    )
    parser.add_argument(
        "--mqtt-host", default="localhost", help="MQTT broker host"
    )
    parser.add_argument(
        "--mqtt-port", type=int, default=1883, help="MQTT broker port"
    )
    parser.add_argument(
        "--site", default="home", help="Tritium site ID"
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level"
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for ``python -m tritium_lib.sdk.examples.weather_station.runner``."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    runner = WeatherStationRunner(
        station_id=args.station,
        poll_interval=args.interval,
        lat=args.lat,
        lng=args.lng,
        site_id=args.site,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
    )

    logger.info("Starting Weather Station Runner: %s", args.station)
    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        logger.info("Shutting down on keyboard interrupt")


if __name__ == "__main__":
    main()
