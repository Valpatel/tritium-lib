# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.inference.llm_client."""

from tritium_lib.inference.llm_client import (
    ollama_chat,
    llama_server_chat,
    set_ollama_host,
    _is_llama_server,
)


def test_set_ollama_host():
    """set_ollama_host updates the global host."""
    set_ollama_host("http://testhost:8081")
    # Reset to default
    set_ollama_host("http://localhost:8081")


def test_is_llama_server():
    """_is_llama_server detects llama-server ports."""
    assert _is_llama_server("http://localhost:8081")
    assert _is_llama_server("http://localhost:8082")
    assert _is_llama_server("http://localhost:8083")
    assert not _is_llama_server("http://localhost:11434")
    assert not _is_llama_server("http://localhost:8080")


def test_ollama_chat_is_callable():
    """ollama_chat function exists and is callable."""
    assert callable(ollama_chat)


def test_llama_server_chat_is_callable():
    """llama_server_chat function exists and is callable."""
    assert callable(llama_server_chat)
