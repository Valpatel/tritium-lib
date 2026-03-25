# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.utils.extraction."""

from tritium_lib.utils.extraction import extract_person_name, extract_facts


def test_extract_person_name_im():
    """Extracts name from 'I'm X' pattern."""
    assert extract_person_name("I'm Matthew") == "Matthew"


def test_extract_person_name_my_name_is():
    """Extracts name from 'my name is X' pattern."""
    assert extract_person_name("my name is Sarah") == "Sarah"


def test_extract_person_name_call_me():
    """Extracts name from 'call me X' pattern."""
    assert extract_person_name("call me Dave") == "Dave"


def test_extract_person_name_none():
    """Returns None when no name found."""
    assert extract_person_name("hello there") is None
    assert extract_person_name("") is None
    assert extract_person_name(None) is None


def test_extract_person_name_stop_word():
    """Filters out common stop words."""
    # "Well" starts with uppercase but is a stop word
    assert extract_person_name("I'm Good") is None


def test_extract_facts_empty():
    """Returns empty list for empty input."""
    assert extract_facts("") == []


def test_extract_facts_schedule():
    """Extracts schedule facts."""
    facts = extract_facts("I have a meeting at 3pm tomorrow")
    assert len(facts) > 0
    assert any("schedule" in f["tags"] for f in facts)


def test_extract_facts_preference():
    """Extracts preference facts."""
    facts = extract_facts("I like programming in Python")
    assert len(facts) > 0
    assert any("preference" in f["tags"] for f in facts)


def test_extract_facts_identity():
    """Extracts identity facts."""
    facts = extract_facts("I'm a software engineer")
    assert len(facts) > 0
    assert any("identity" in f["tags"] for f in facts)


def test_extract_facts_with_person():
    """Facts include person when provided."""
    facts = extract_facts("I like coffee", person="Matt")
    for f in facts:
        assert f["person"] == "Matt"


def test_extract_facts_possession():
    """Extracts possession facts."""
    facts = extract_facts("I have a new laptop computer")
    assert any("possession" in f["tags"] for f in facts)
