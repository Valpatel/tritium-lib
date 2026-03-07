# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""MQTT topic conventions and codecs for the Tritium ecosystem.

All Tritium services use the same topic hierarchy:
  tritium/{site}/{domain}/{device_id}/{data_type}

This module defines the topic patterns and JSON codecs so that
tritium-sc and tritium-edge speak the same language.
"""

from .topics import TritiumTopics

__all__ = ["TritiumTopics"]
