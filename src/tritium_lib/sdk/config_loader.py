# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""AddonConfig — runtime configuration loaded from addon manifests.

Reads the ``[config]`` section from an addon manifest, applies schema
defaults, then overlays any runtime overrides.
"""

from __future__ import annotations

from typing import Any


class AddonConfig:
    """Runtime config for an addon, loaded from manifest [config] section."""

    def __init__(
        self,
        config_schema: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
    ):
        self._schema = config_schema or {}
        self._values: dict[str, Any] = {}
        # Apply defaults from schema
        for key, spec in self._schema.items():
            if isinstance(spec, dict):
                self._values[key] = spec.get("default")
            else:
                self._values[key] = spec
        # Apply overrides
        if overrides:
            self._values.update(overrides)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by key."""
        return self._values.get(key, default)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._values.get(name)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict copy of all config values."""
        return dict(self._values)

    def __repr__(self) -> str:
        return f"AddonConfig({self._values!r})"
