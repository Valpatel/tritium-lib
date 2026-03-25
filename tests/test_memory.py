# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.utils.memory."""

import os
import tempfile
from tritium_lib.utils.memory import Memory


def test_memory_instantiation():
    """Memory can be created with custom path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_memory.json")
        m = Memory(path=path)
        assert m is not None
        assert m.session_count == 1


def test_memory_add_event():
    """Memory records events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        m = Memory(path=os.path.join(tmpdir, "mem.json"))
        m.add_event("test", "something happened")
        events = m.get_recent_events(10)
        assert len(events) == 1
        assert events[0]["type"] == "test"


def test_memory_add_observation():
    """Memory records spatial observations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        m = Memory(path=os.path.join(tmpdir, "mem.json"))
        m.add_observation(0.0, 0.0, "desk with monitor")
        nearby = m.get_nearby_observations(0.0, 0.0)
        assert len(nearby) == 1
        assert "desk" in nearby[0]


def test_memory_save_load():
    """Memory persists to disk and reloads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mem.json")
        m = Memory(path=path)
        m.add_event("test", "hello")
        m.add_fact("I like coffee", tags=["preference"])
        m.save()

        m2 = Memory(path=path)
        events = m2.get_recent_events(10)
        assert len(events) == 1
        assert m2.session_count == 2  # Incremented on load


def test_memory_add_fact():
    """Memory stores facts with caps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        m = Memory(path=os.path.join(tmpdir, "mem.json"))
        m.add_fact("test fact", tags=["test"], person="Alice")
        assert len(m.facts) == 1
        assert m.facts[0]["person"] == "Alice"


def test_memory_link_person():
    """Memory tracks known people."""
    with tempfile.TemporaryDirectory() as tmpdir:
        m = Memory(path=os.path.join(tmpdir, "mem.json"))
        m.link_person("Alice", appearance="tall", zone="office")
        assert "alice" in m.known_people
        assert m.known_people["alice"]["name"] == "Alice"


def test_memory_recall():
    """Memory recall searches facts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        m = Memory(path=os.path.join(tmpdir, "mem.json"))
        m.add_fact("Alice likes coffee")
        m.add_fact("Bob drives a truck")
        results = m.recall("coffee")
        assert len(results) > 0
        assert "coffee" in results[0]["text"].lower()


def test_memory_zones():
    """Memory registers and finds zones."""
    with tempfile.TemporaryDirectory() as tmpdir:
        m = Memory(path=os.path.join(tmpdir, "mem.json"))
        m.register_zone("office", pan=10.0, tilt=5.0, description="Main office")
        zone = m.get_zone_at(12.0, 6.0)
        assert zone is not None
        assert zone["name"] == "office"


def test_memory_preferences():
    """Memory tracks preferences with dedup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        m = Memory(path=os.path.join(tmpdir, "mem.json"))
        m.add_preference("likes", "coffee")
        m.add_preference("likes", "coffee")  # Duplicate
        m.add_preference("likes", "tea")
        likes = m.self_model["preferences"]["likes"]
        assert len(likes) == 2  # Deduped


def test_memory_dashboard_data():
    """Memory returns dashboard data dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        m = Memory(path=os.path.join(tmpdir, "mem.json"))
        data = m.get_dashboard_data()
        assert "session" in data
        assert "uptime_min" in data
        assert "total_events" in data
