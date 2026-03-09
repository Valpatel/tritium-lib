# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium-lib — Shared library for the Tritium ecosystem.

Provides reusable components across tritium-sc and tritium-edge:
  - tritium_lib.events   — Event bus interface
  - tritium_lib.mqtt     — MQTT topic conventions and codecs
  - tritium_lib.config   — Pydantic settings base classes
  - tritium_lib.models   — Shared data models (Device, Sensor, Command)
  - tritium_lib.store    — Shared persistence (BLE sighting store)
  - tritium_lib.auth     — JWT utilities
  - tritium_lib.web      — Web UI theme, components, dashboards, templates
"""

__version__ = "0.1.0"
