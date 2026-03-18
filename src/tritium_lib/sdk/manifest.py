# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""Addon manifest parser and validator.

Reads tritium_addon.toml files and validates required fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # fallback
    except ImportError:
        tomllib = None  # type: ignore


@dataclass
class AddonManifest:
    """Parsed addon manifest from tritium_addon.toml."""

    # Required
    id: str = ""
    name: str = ""
    version: str = "0.0.0"

    # Optional metadata
    description: str = ""
    author: str = ""
    license: str = "AGPL-3.0"
    addon_api: str = ">=1.0, <2.0"

    # Category / UI
    category_window: str = "system"
    category_tab_order: int = 99
    category_icon: str = ""

    # Dependencies
    requires: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)
    python_packages: list[str] = field(default_factory=list)

    # Hardware
    hardware_devices: list[str] = field(default_factory=list)
    serial_vid_pid: list[str] = field(default_factory=list)
    auto_detect: bool = False

    # Permissions
    perm_serial: bool = False
    perm_network: bool = False
    perm_mqtt: bool = False
    perm_storage: bool = False

    # Backend
    module: str = ""
    router_prefix: str = ""
    mqtt_topics: list[str] = field(default_factory=list)

    # Frontend
    panels: list[dict] = field(default_factory=list)
    layers: list[dict] = field(default_factory=list)
    context_menu: list[dict] = field(default_factory=list)
    shortcuts: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)

    # Config fields
    config_fields: dict = field(default_factory=dict)

    # Source path (where the manifest was loaded from)
    path: Optional[Path] = None

    def to_frontend_json(self) -> dict:
        """Return a JSON-serializable dict for the frontend addon loader."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "category": self.category_window,
            "icon": self.category_icon,
            "panels": self.panels,
            "layers": self.layers,
            "context_menu": self.context_menu,
            "shortcuts": self.shortcuts,
            "tools": self.tools,
        }


def load_manifest(path: str | Path) -> AddonManifest:
    """Load and parse a tritium_addon.toml file.

    Args:
        path: Path to the tritium_addon.toml file.

    Returns:
        Parsed AddonManifest.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required fields are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    if tomllib is None:
        raise ImportError("tomllib or tomli required to parse TOML manifests")

    with open(path, "rb") as f:
        data = tomllib.load(f)

    manifest = AddonManifest(path=path.parent)

    # [addon] section — required
    addon = data.get("addon", {})
    manifest.id = addon.get("id", "")
    manifest.name = addon.get("name", "")
    manifest.version = addon.get("version", "0.0.0")
    manifest.description = addon.get("description", "")
    manifest.author = addon.get("author", "")
    manifest.license = addon.get("license", "AGPL-3.0")
    manifest.addon_api = addon.get("addon_api", ">=1.0, <2.0")

    # [addon.category]
    cat = addon.get("category", {})
    if isinstance(cat, str):
        manifest.category_window = cat
    elif isinstance(cat, dict):
        manifest.category_window = cat.get("window", "system")
        manifest.category_tab_order = cat.get("tab_order", 99)
        manifest.category_icon = cat.get("icon", "")

    # [dependencies]
    deps = data.get("dependencies", {})
    manifest.requires = deps.get("requires", [])
    manifest.optional = deps.get("optional", [])
    manifest.python_packages = deps.get("python_packages", [])

    # [hardware]
    hw = data.get("hardware", {})
    manifest.hardware_devices = hw.get("devices", [])
    manifest.serial_vid_pid = hw.get("serial_vid_pid", [])
    manifest.auto_detect = hw.get("auto_detect", False)

    # [permissions]
    perms = data.get("permissions", {})
    manifest.perm_serial = perms.get("serial", False)
    manifest.perm_network = perms.get("network", False)
    manifest.perm_mqtt = perms.get("mqtt", False)
    manifest.perm_storage = perms.get("storage", False)

    # [backend]
    backend = data.get("backend", {})
    manifest.module = backend.get("module", "")
    manifest.router_prefix = backend.get("router_prefix", f"/api/addons/{manifest.id}")
    manifest.mqtt_topics = backend.get("mqtt_topics", [])

    # [frontend]
    frontend = data.get("frontend", {})
    manifest.panels = frontend.get("panels", [])
    manifest.layers = frontend.get("layers", [])
    manifest.context_menu = frontend.get("context_menu", [])
    manifest.shortcuts = frontend.get("shortcuts", [])
    manifest.tools = frontend.get("tools", [])

    # [config]
    manifest.config_fields = data.get("config", {})

    return manifest


def validate_manifest(manifest: AddonManifest) -> list[str]:
    """Validate a parsed manifest. Returns list of error messages (empty = valid).

    Args:
        manifest: Parsed AddonManifest to validate.

    Returns:
        List of error strings. Empty list means valid.
    """
    errors = []

    if not manifest.id:
        errors.append("Missing required field: addon.id")
    if not manifest.name:
        errors.append("Missing required field: addon.name")
    if not manifest.version:
        errors.append("Missing required field: addon.version")

    # ID format: lowercase, hyphens, no spaces
    if manifest.id and not all(c.isalnum() or c == '-' for c in manifest.id):
        errors.append(f"Invalid addon.id '{manifest.id}': use lowercase letters, numbers, hyphens only")

    # Panels must have id and title
    for i, panel in enumerate(manifest.panels):
        if not panel.get("id"):
            errors.append(f"Panel {i} missing 'id'")
        if not panel.get("title"):
            errors.append(f"Panel {i} missing 'title'")

    # Layers must have id and label
    for i, layer in enumerate(manifest.layers):
        if not layer.get("id"):
            errors.append(f"Layer {i} missing 'id'")
        if not layer.get("label"):
            errors.append(f"Layer {i} missing 'label'")

    return errors
