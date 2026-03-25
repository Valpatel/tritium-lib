# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.inference.fleet."""

from tritium_lib.inference.fleet import LLMFleet, FleetHost, OllamaFleet


def test_fleet_host_dataclass():
    """FleetHost can be created with required fields."""
    h = FleetHost(url="http://localhost:8081", name="localhost")
    assert h.url == "http://localhost:8081"
    assert h.name == "localhost"
    assert h.models == []
    assert h.backend == "llama-server"


def test_fleet_host_has_model():
    """FleetHost.has_model does prefix matching."""
    h = FleetHost(url="http://localhost:8081", name="localhost", models=["qwen2.5:7b"])
    assert h.has_model("qwen2.5:7b")
    assert h.has_model("qwen2.5")
    assert not h.has_model("llama3")


def test_fleet_instantiation_no_discover():
    """LLMFleet can be created with auto_discover=False."""
    fleet = LLMFleet(auto_discover=False)
    assert fleet is not None
    assert isinstance(fleet.count, int)


def test_fleet_has_discovery_methods():
    """LLMFleet has hosts_with_model and best_host methods."""
    fleet = LLMFleet(auto_discover=False)
    assert hasattr(fleet, "hosts_with_model")
    assert hasattr(fleet, "best_host")
    assert hasattr(fleet, "refresh")
    assert hasattr(fleet, "chat")
    assert hasattr(fleet, "generate")
    assert hasattr(fleet, "status")


def test_fleet_hosts_property():
    """LLMFleet.hosts returns a list."""
    fleet = LLMFleet(auto_discover=False)
    assert isinstance(fleet.hosts, list)


def test_fleet_status_no_hosts():
    """LLMFleet.status reports no hosts when none found."""
    fleet = LLMFleet(auto_discover=False)
    status = fleet.status()
    assert "0 host" in status or "host" in status


def test_backward_compat_alias():
    """OllamaFleet is an alias for LLMFleet."""
    assert OllamaFleet is LLMFleet


def test_fleet_host_has_model_empty_models():
    """FleetHost.has_model returns False when models list is empty."""
    h = FleetHost(url="http://localhost:8081", name="localhost", models=[])
    assert not h.has_model("anything")


def test_fleet_host_has_model_with_tag():
    """FleetHost.has_model splits on colon for prefix match."""
    h = FleetHost(url="http://localhost:8081", name="localhost", models=["llama3:8b-q4"])
    assert h.has_model("llama3:8b-q4")
    assert h.has_model("llama3")
    assert not h.has_model("qwen")


def test_fleet_best_host_no_hosts():
    """best_host returns None when fleet is empty."""
    fleet = LLMFleet(auto_discover=False)
    fleet._hosts.clear()
    assert fleet.best_host("any-model") is None


def test_fleet_hosts_with_model_empty():
    """hosts_with_model returns empty list for unknown model."""
    fleet = LLMFleet(auto_discover=False)
    fleet._hosts.clear()
    assert fleet.hosts_with_model("nonexistent") == []


def test_fleet_chat_no_hosts_returns_empty():
    """chat returns empty string when no hosts available."""
    fleet = LLMFleet(auto_discover=False)
    fleet._hosts.clear()
    result = fleet.chat("model", "prompt")
    assert result == ""


def test_fleet_generate_no_hosts_returns_empty():
    """generate returns empty string when no hosts available."""
    fleet = LLMFleet(auto_discover=False)
    fleet._hosts.clear()
    result = fleet.generate("model", "prompt")
    assert result == ""
