# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""MQTT topic conventions — single source of truth for topic patterns.

Both tritium-sc and tritium-edge import these to ensure consistency.
"""


class TritiumTopics:
    """Topic builder for Tritium MQTT messages."""

    def __init__(self, site_id: str = "home"):
        self.site = site_id
        self.prefix = f"tritium/{site_id}"

    # --- Edge device topics (tritium-edge) ---

    def edge_heartbeat(self, device_id: str) -> str:
        return f"{self.prefix}/edge/{device_id}/heartbeat"

    def edge_telemetry(self, device_id: str) -> str:
        return f"{self.prefix}/edge/{device_id}/telemetry"

    def edge_command(self, device_id: str) -> str:
        return f"{self.prefix}/edge/{device_id}/command"

    def edge_ota_status(self, device_id: str) -> str:
        return f"{self.prefix}/edge/{device_id}/ota"

    # --- Sensor topics (shared) ---

    def sensor(self, device_id: str, sensor_type: str) -> str:
        return f"{self.prefix}/sensors/{device_id}/{sensor_type}"

    def sensor_wildcard(self, device_id: str = "+") -> str:
        return f"{self.prefix}/sensors/{device_id}/#"

    # --- Camera topics (shared) ---

    def camera_frame(self, device_id: str) -> str:
        return f"{self.prefix}/cameras/{device_id}/frame"

    def camera_detections(self, device_id: str) -> str:
        return f"{self.prefix}/cameras/{device_id}/detections"

    def camera_command(self, device_id: str) -> str:
        return f"{self.prefix}/cameras/{device_id}/command"

    # --- Audio topics (shared) ---

    def audio_stream(self, device_id: str) -> str:
        return f"{self.prefix}/audio/{device_id}/stream"

    def audio_vad(self, device_id: str) -> str:
        return f"{self.prefix}/audio/{device_id}/vad"

    # --- Mesh topics ---

    def mesh_peers(self, device_id: str) -> str:
        return f"{self.prefix}/mesh/{device_id}/peers"

    # --- Robot topics (tritium-sc) ---

    def robot_telemetry(self, robot_id: str) -> str:
        return f"{self.prefix}/robots/{robot_id}/telemetry"

    def robot_command(self, robot_id: str) -> str:
        return f"{self.prefix}/robots/{robot_id}/command"

    def robot_thoughts(self, robot_id: str) -> str:
        return f"{self.prefix}/robots/{robot_id}/thoughts"

    # --- System topics ---

    def alerts(self) -> str:
        return f"{self.prefix}/amy/alerts"

    def escalation(self) -> str:
        return f"{self.prefix}/escalation/change"

    # --- Wildcards for subscriptions ---

    def all_edge(self) -> str:
        return f"{self.prefix}/edge/+/#"

    def all_sensors(self) -> str:
        return f"{self.prefix}/sensors/+/#"

    def all_cameras(self) -> str:
        return f"{self.prefix}/cameras/+/#"
